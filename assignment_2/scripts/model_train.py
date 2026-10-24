import argparse
import pyspark

import utils.model_train

# to call this script: python scripts/model_train.py --snapshotdate "2024-06-01"


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # run training: train candidates, select best, save to model bank
    utils.model_train.train_model(
        snapshotdate,
        gold_feature_store_directory="datamart/gold/feature_store/",
        gold_label_store_directory="datamart/gold/label_store/",
        model_bank_directory="model_bank/",
        spark=spark,
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (model training date)")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
