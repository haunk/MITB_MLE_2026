import os
import glob
import pickle
import pprint
from datetime import datetime
from dateutil.relativedelta import relativedelta

from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, brier_score_loss

import xgboost as xgb


# non-feature columns in the gold feature store / joined data
ID_COLS = ["Customer_ID", "loan_id", "snapshot_date", "label"]


def train_model(
    model_train_date_str,
    gold_feature_store_directory,
    gold_label_store_directory,
    model_bank_directory,
    spark,
    label_delay_months=6,
    train_test_period_months=10,
    oot_period_months=2,
    train_test_ratio=0.8,
):
    """
    Training pipeline (based on lab 4/5 model_train_main.ipynb).
    Trains 2 candidate models, selects the best on OOT AUC and saves it to the model bank.

    Design decisions:
        1. Time-based split on application month (feature snapshot_date).
           A label is only observed label_delay_months (6) after application (30dpd_6mob),
           so at model_train_date only applications up to model_train_date - 6 months
           have labels. Using later labels would be temporal leakage.
           e.g. model_train_date 2024-06-01:
               train/test: 2023-01 to 2023-10 (80/20 stratified split)
               OOT:        2023-11 to 2023-12
        2. Imputer (median) and scaler are fitted on the train split only (no train-test
           contamination) and saved in the artefact for inference.
        3. Candidates: Logistic Regression (simple, explainable baseline) and
           XGBoost (tuned with random search, as in lab). Best = highest OOT AUC.
        4. Calibration (lecture 4): the selected model's scores are calibrated with a sigmoid
           (Platt) calibrator fitted on the test split (not train, to avoid overfitting), so the
           output can be used as a probability of default (PD). Ranking (AUC) is unchanged.
           Brier score on OOT before/after calibration is recorded.
        5. Artefact also stores feature_cols (inference uses the same columns), the calibrator,
           the calibrated train-set scores as the baseline distribution for PSI monitoring, and
           the train-set values of the top 5 most important features for CSI monitoring.

    Args:
        model_train_date_str: training date 'YYYY-MM-DD'
        gold_feature_store_directory: path to gold feature store partitions
        gold_label_store_directory: path to gold label store partitions
        model_bank_directory: path to model bank output directory
        spark: SparkSession

    Returns:
        dict model artefact of the selected model
    """
    # --- set up config ---
    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    # latest application month whose label is already observed at training date
    config["oot_end_date"] = config["model_train_date"] - relativedelta(months=label_delay_months)
    config["oot_start_date"] = config["oot_end_date"] - relativedelta(months=oot_period_months - 1)
    config["train_test_end_date"] = config["oot_start_date"] - relativedelta(months=1)
    config["train_test_start_date"] = config["train_test_end_date"] - relativedelta(months=train_test_period_months - 1)
    config["train_test_ratio"] = train_test_ratio

    pprint.pprint(config)

    # --- get label store: only labels observed up to training date ---
    files_list = [gold_label_store_directory + os.path.basename(f) for f in glob.glob(os.path.join(gold_label_store_directory, '*'))]
    label_store_sdf = spark.read.parquet(*files_list)
    labels_sdf = label_store_sdf.filter(col("snapshot_date") <= config["model_train_date"]).select("loan_id", "label")
    print("extracted labels_sdf", labels_sdf.count())

    # --- get feature store: applications in train/test + OOT period ---
    files_list = [gold_feature_store_directory + os.path.basename(f) for f in glob.glob(os.path.join(gold_feature_store_directory, '*'))]
    feature_store_sdf = spark.read.parquet(*files_list)
    features_sdf = feature_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"])
    )
    print("extracted features_sdf", features_sdf.count())

    # --- prepare data for modeling: feature (application) joined to label on loan_id ---
    data_pdf = features_sdf.join(labels_sdf, on="loan_id", how="inner").toPandas()

    # split data into train - test - oot
    oot_pdf = data_pdf[(data_pdf["snapshot_date"] >= config["oot_start_date"].date()) & (data_pdf["snapshot_date"] <= config["oot_end_date"].date())]
    train_test_pdf = data_pdf[(data_pdf["snapshot_date"] >= config["train_test_start_date"].date()) & (data_pdf["snapshot_date"] <= config["train_test_end_date"].date())]

    feature_cols = [c for c in data_pdf.columns if c not in ID_COLS]

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"],
        test_size=1 - config["train_test_ratio"],
        random_state=88,     # Ensures reproducibility
        shuffle=True,        # Shuffle the data before splitting
        stratify=train_test_pdf["label"]           # Stratify based on the label column
    )

    print('X_train', X_train.shape[0], '| X_test', X_test.shape[0], '| X_oot', X_oot.shape[0], '| features', len(feature_cols))
    print('bad rate train', round(y_train.mean(), 3), '| test', round(y_test.mean(), 3), '| oot', round(y_oot.mean(), 3))

    # --- preprocess data: fit on train only ---
    transformer_imputer = SimpleImputer(strategy="median").fit(X_train)
    transformer_stdscaler = StandardScaler().fit(transformer_imputer.transform(X_train))

    def preprocess(X):
        return transformer_stdscaler.transform(transformer_imputer.transform(X))

    X_train_processed = preprocess(X_train)
    X_test_processed = preprocess(X_test)
    X_oot_processed = preprocess(X_oot)

    # --- train candidate models ---
    candidates = {}

    # candidate 1: logistic regression baseline
    lr_clf = LogisticRegression(max_iter=1000, random_state=88)
    lr_clf.fit(X_train_processed, y_train)
    candidates["logistic_regression"] = {"model": lr_clf, "hp_params": lr_clf.get_params()}

    # candidate 2: xgboost with random search (hyperparameter space from lab)
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring="roc_auc",  # AUC on predicted probabilities
        n_iter=50,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train_processed, y_train)
    candidates["xgboost"] = {"model": random_search.best_estimator_, "hp_params": random_search.best_params_}

    # --- evaluate candidates on train / test / oot ---
    results = {}
    for name, candidate in candidates.items():
        model = candidate["model"]
        auc_train = roc_auc_score(y_train, model.predict_proba(X_train_processed)[:, 1])
        auc_test = roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])
        auc_oot = roc_auc_score(y_oot, model.predict_proba(X_oot_processed)[:, 1])
        results[name] = {
            'auc_train': auc_train,
            'auc_test': auc_test,
            'auc_oot': auc_oot,
            'gini_train': round(2 * auc_train - 1, 3),
            'gini_test': round(2 * auc_test - 1, 3),
            'gini_oot': round(2 * auc_oot - 1, 3),
        }
    pprint.pprint(results)

    # --- select best model on OOT AUC ---
    best_model_type = max(results, key=lambda name: results[name]["auc_oot"])
    best_model = candidates[best_model_type]["model"]
    print("selected model:", best_model_type)

    # --- calibrate selected model: sigmoid (Platt) calibrator fitted on test split ---
    calibrator = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid")
    calibrator.fit(X_test_processed, y_test)

    y_oot_raw = best_model.predict_proba(X_oot_processed)[:, 1]
    y_oot_calibrated = calibrator.predict_proba(X_oot_processed)[:, 1]
    calibration_results = {
        'bad_rate_oot': round(y_oot.mean(), 3),
        'mean_pred_oot_raw': round(y_oot_raw.mean(), 3),
        'mean_pred_oot_calibrated': round(y_oot_calibrated.mean(), 3),
        'brier_oot_raw': round(brier_score_loss(y_oot, y_oot_raw), 4),
        'brier_oot_calibrated': round(brier_score_loss(y_oot, y_oot_calibrated), 4),
        'auc_oot_calibrated': roc_auc_score(y_oot, y_oot_calibrated),
    }
    pprint.pprint(calibration_results)

    # --- prepare model artefact to save ---
    model_artefact = {}
    model_artefact['model'] = best_model
    model_artefact['calibrator'] = calibrator  # wraps model; use calibrator.predict_proba for PD
    model_artefact['model_version'] = "credit_model_" + config["model_train_date_str"].replace('-', '_')
    model_artefact['model_type'] = best_model_type
    model_artefact['feature_cols'] = feature_cols
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['imputer'] = transformer_imputer
    model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(), 2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(), 2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(), 2)
    model_artefact['results'] = results[best_model_type]
    model_artefact['candidate_results'] = results
    model_artefact['calibration_results'] = calibration_results
    model_artefact['hp_params'] = candidates[best_model_type]["hp_params"]
    # baseline (calibrated) score distribution for PSI monitoring, same scale as inference output
    model_artefact['baseline_scores'] = calibrator.predict_proba(X_train_processed)[:, 1]
    # top 5 most important features and their raw train values for CSI monitoring
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_  # xgboost
    else:
        importances = abs(best_model.coef_[0])  # logistic regression (features are scaled)
    top_features = [f for _, f in sorted(zip(importances, feature_cols), reverse=True)[:5]]
    model_artefact['csi_features'] = top_features
    model_artefact['baseline_features'] = X_train[top_features].reset_index(drop=True)
    print("top 5 features for CSI:", top_features)

    # --- save artefact to model bank ---
    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)

    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to {file_path}")

    return model_artefact
