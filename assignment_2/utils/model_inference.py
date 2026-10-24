import os
import pickle


def model_inference(
    snapshot_date_str,
    model_bank_directory,
    model_name,
    gold_feature_store_directory,
    gold_predictions_directory,
    spark,
):
    """
    Inference pipeline (based on lab 5 model_inference.py).
    Loads the model artefact from the model bank, scores the loan applications of
    snapshot_date and saves the predictions as a gold table.

    Design decisions:
        1. Uses the artefact's feature_cols, imputer and scaler (fitted at training) so
           inference applies exactly the same preprocessing as training (no training/serving skew).
           Output is the calibrated probability of default (artefact's calibrator).
        2. Batch inference once a month on new applications (mob == 0 rows in gold feature store).
        3. Months without new applications have no feature store partition: skip.

    Args:
        snapshot_date_str: date string in 'YYYY-MM-DD' format
        model_bank_directory: path to model bank
        model_name: model artefact file name, e.g. 'credit_model_2024_06_01.pkl'
        gold_feature_store_directory: path to gold feature store partitions
        gold_predictions_directory: path to gold model predictions output directory
        spark: SparkSession

    Returns:
        pandas DataFrame of predictions, or None if no applications this month
    """
    date_tag = snapshot_date_str.replace("-", "_")

    # --- load model artefact from model bank ---
    model_artefact_filepath = os.path.join(model_bank_directory, model_name)
    with open(model_artefact_filepath, 'rb') as file:
        model_artefact = pickle.load(file)

    print("Model loaded successfully! " + model_artefact_filepath)

    # --- load feature store partition for this month ---
    feature_file = os.path.join(gold_feature_store_directory, f"gold_feature_store_{date_tag}.parquet")
    if not os.path.exists(feature_file):
        print(f"{snapshot_date_str} | no feature store partition (no new applications), skipping")
        return None

    features_pdf = spark.read.parquet(feature_file).toPandas()
    print("extracted features", features_pdf.shape[0], snapshot_date_str)

    # --- preprocess data for modeling: same columns and transformers as training ---
    X_inference = features_pdf[model_artefact["feature_cols"]]
    X_inference = model_artefact["preprocessing_transformers"]["imputer"].transform(X_inference)
    X_inference = model_artefact["preprocessing_transformers"]["stdscaler"].transform(X_inference)

    # --- model prediction inference: calibrated probability of default ---
    calibrator = model_artefact["calibrator"]
    y_inference = calibrator.predict_proba(X_inference)[:, 1]

    # prepare output
    y_inference_pdf = features_pdf[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = model_artefact["model_version"]
    y_inference_pdf["model_predictions"] = y_inference

    # --- save model inference to datamart gold table ---
    gold_directory = os.path.join(gold_predictions_directory, model_artefact["model_version"]) + "/"
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    partition_name = model_artefact["model_version"] + "_predictions_" + date_tag + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, '| rows:', y_inference_pdf.shape[0])

    return y_inference_pdf
