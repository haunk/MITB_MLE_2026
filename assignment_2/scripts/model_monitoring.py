import argparse
import pyspark

import utils.model_monitoring

# to call this script: python scripts/model_monitoring.py --snapshotdate "2024-12-01" --modelname "credit_model_2024_06_01.pkl"


def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # run monitoring: stability (this month) + performance (labels arriving this month)
    utils.model_monitoring.model_monitoring(
        snapshotdate,
        model_bank_directory="model_bank/",
        model_name=modelname,
        gold_predictions_directory="datamart/gold/model_predictions/",
        gold_feature_store_directory="datamart/gold/feature_store/",
        gold_label_store_directory="datamart/gold/label_store/",
        gold_monitoring_directory="datamart/gold/model_monitoring/",
        spark=spark,
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model artefact file name in model_bank/")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
