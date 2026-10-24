import os
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score


# governance thresholds (green / amber / red)
PSI_THRESHOLDS = (0.10, 0.25)   # lecture 7: <0.1 no drift, 0.1-0.25 moderate, >0.25 significant
AUC_THRESHOLDS = (0.70, 0.75)   # >=0.75 green, 0.70-0.75 amber, <0.70 red
GINI_THRESHOLDS = (0.40, 0.50)  # same as AUC thresholds (gini = 2*auc - 1)
GAP_THRESHOLDS = (0.03, 0.05)   # |predicted - actual bad rate|: <0.03 green, 0.03-0.05 amber, >0.05 red


def calculate_psi(expected, actual, bins=10):
    """
    Population Stability Index between a baseline (expected) and current (actual) distribution.
    Bins are deciles of the baseline. Also used as CSI when applied to a single feature.
    Missing values are counted as their own bin, so a change in missing rate
    (e.g. an upstream source stops sending data) shows up as drift.
    """
    expected = pd.Series(expected, dtype=float)
    actual = pd.Series(actual, dtype=float)
    if expected.notna().sum() == 0 or len(actual) == 0:
        return None

    # bin edges from baseline quantiles (unique: discrete features have fewer bins)
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    # share per bin, plus one extra bin for missing values
    expected_pct = np.append(np.histogram(expected.dropna(), edges)[0], expected.isna().sum()) / len(expected)
    actual_pct = np.append(np.histogram(actual.dropna(), edges)[0], actual.isna().sum()) / len(actual)

    # avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def get_status(value, thresholds, higher_is_better):
    """Traffic light status for a metric value given (low, high) thresholds."""
    if value is None:
        return "n/a"
    low, high = thresholds
    if higher_is_better:
        return "green" if value >= high else ("amber" if value >= low else "red")
    return "green" if value < low else ("amber" if value <= high else "red")


def model_monitoring(
    snapshot_date_str,
    model_bank_directory,
    model_name,
    gold_predictions_directory,
    gold_feature_store_directory,
    gold_label_store_directory,
    gold_monitoring_directory,
    spark,
    label_delay_months=6,
    cutoff=0.5,
):
    """
    Monitoring pipeline (lecture 7): stability and performance of the model over time.
    Saves one partition per monitoring date as a gold table in long format
    (one row per metric).

    Design decisions:
        1. Stability (available immediately) for predictions made in snapshot_date:
           - PSI of predicted PD vs training (baseline) score distribution
           - CSI of the top 5 features vs their training distribution
           - number of predictions, mean predicted PD
        2. Performance (available label_delay_months later) for predictions made in
           snapshot_date - 6 months, whose 30dpd_6mob labels arrive in snapshot_date:
           - AUC, Gini, Brier, actual vs mean predicted bad rate (and their gap)
           - precision, recall, F1 at PD cutoff (default 0.5)
        3. Each metric gets a status (green / amber / red) from governance thresholds.

    Args:
        snapshot_date_str: monitoring date 'YYYY-MM-DD'
        model_bank_directory: path to model bank
        model_name: model artefact file name, e.g. 'credit_model_2024_06_01.pkl'
        gold_predictions_directory: path to gold model predictions
        gold_feature_store_directory: path to gold feature store partitions
        gold_label_store_directory: path to gold label store partitions
        gold_monitoring_directory: path to gold model monitoring output directory
        spark: SparkSession

    Returns:
        pandas DataFrame of monitoring metrics, or None if nothing to monitor
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    performance_date = snapshot_date - relativedelta(months=label_delay_months)

    # --- load model artefact from model bank ---
    with open(os.path.join(model_bank_directory, model_name), 'rb') as file:
        model_artefact = pickle.load(file)
    model_version = model_artefact["model_version"]
    predictions_directory = os.path.join(gold_predictions_directory, model_version)

    rows = []

    def add_row(prediction_date, metric_type, metric_name, value, status="n/a"):
        rows.append({
            "model_name": model_version,
            "monitoring_date": snapshot_date.date(),
            "prediction_month": prediction_date.date(),
            "metric_type": metric_type,
            "metric_name": metric_name,
            "metric_value": None if value is None else float(value),
            "status": status,
        })

    # ------------------------------------------------------------------ #
    # 1. Stability: predictions made this month
    # ------------------------------------------------------------------ #
    date_tag = snapshot_date_str.replace("-", "_")
    pred_file = os.path.join(predictions_directory, f"{model_version}_predictions_{date_tag}.parquet")

    if os.path.exists(pred_file):
        pred_pdf = spark.read.parquet(pred_file).toPandas()

        psi = calculate_psi(model_artefact["baseline_scores"], pred_pdf["model_predictions"])
        add_row(snapshot_date, "stability", "n_predictions", len(pred_pdf))
        add_row(snapshot_date, "stability", "mean_pd", pred_pdf["model_predictions"].mean())
        add_row(snapshot_date, "stability", "psi_score", psi, get_status(psi, PSI_THRESHOLDS, higher_is_better=False))

        # CSI on top features (raw values from gold feature store)
        feature_file = os.path.join(gold_feature_store_directory, f"gold_feature_store_{date_tag}.parquet")
        features_pdf = spark.read.parquet(feature_file).toPandas()
        for feature in model_artefact["csi_features"]:
            csi = calculate_psi(model_artefact["baseline_features"][feature], features_pdf[feature])
            add_row(snapshot_date, "stability", f"csi_{feature}", csi, get_status(csi, PSI_THRESHOLDS, higher_is_better=False))
        print(f"{snapshot_date_str} | stability: n={len(pred_pdf)} | psi={psi:.4f}")
    else:
        print(f"{snapshot_date_str} | stability: no predictions this month")

    # ------------------------------------------------------------------ #
    # 2. Performance: predictions made label_delay_months ago, labels arrive now
    # ------------------------------------------------------------------ #
    perf_tag = performance_date.strftime("%Y_%m_%d")
    perf_pred_file = os.path.join(predictions_directory, f"{model_version}_predictions_{perf_tag}.parquet")
    label_file = os.path.join(gold_label_store_directory, f"gold_label_store_{date_tag}.parquet")

    if os.path.exists(perf_pred_file) and os.path.exists(label_file):
        perf_pred_pdf = spark.read.parquet(perf_pred_file).toPandas()
        labels_pdf = spark.read.parquet(label_file).toPandas()
        perf_pdf = perf_pred_pdf.merge(labels_pdf[["loan_id", "label"]], on="loan_id", how="inner")

        if len(perf_pdf) > 0:
            y_true = perf_pdf["label"]
            y_score = perf_pdf["model_predictions"]
            y_pred = (y_score >= cutoff).astype(int)

            auc = roc_auc_score(y_true, y_score)
            gini = 2 * auc - 1
            add_row(performance_date, "performance", "n_labels", len(perf_pdf))
            add_row(performance_date, "performance", "bad_rate_actual", y_true.mean())
            add_row(performance_date, "performance", "bad_rate_predicted", y_score.mean())
            gap = y_score.mean() - y_true.mean()  # negative = model under-predicts default risk
            add_row(performance_date, "performance", "bad_rate_gap", gap, get_status(abs(gap), GAP_THRESHOLDS, higher_is_better=False))
            add_row(performance_date, "performance", "auc", auc, get_status(auc, AUC_THRESHOLDS, higher_is_better=True))
            add_row(performance_date, "performance", "gini", gini, get_status(gini, GINI_THRESHOLDS, higher_is_better=True))
            add_row(performance_date, "performance", "brier", brier_score_loss(y_true, y_score))
            add_row(performance_date, "performance", "precision", precision_score(y_true, y_pred, zero_division=0))
            add_row(performance_date, "performance", "recall", recall_score(y_true, y_pred, zero_division=0))
            add_row(performance_date, "performance", "f1", f1_score(y_true, y_pred, zero_division=0))
            print(f"{snapshot_date_str} | performance of {performance_date.date()}: n={len(perf_pdf)} | auc={auc:.4f}")
    else:
        print(f"{snapshot_date_str} | performance: no predictions with labels arriving this month")

    if not rows:
        print(f"{snapshot_date_str} | nothing to monitor, skipping")
        return None

    # --- save monitoring results to datamart gold table ---
    monitoring_pdf = pd.DataFrame(rows)

    gold_directory = os.path.join(gold_monitoring_directory, model_version) + "/"
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    filepath = gold_directory + f"{model_version}_monitoring_{date_tag}.parquet"
    spark.createDataFrame(monitoring_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, '| rows:', len(monitoring_pdf))

    return monitoring_pdf
