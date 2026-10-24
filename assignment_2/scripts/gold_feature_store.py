import argparse
import os
import pyspark

import utils.data_processing_gold_table

# to call this script: python scripts/gold_feature_store.py --snapshotdate "2023-01-01"


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # create gold datalake
    silver_loan_directory = "datamart/silver/loan/"
    silver_attributes_directory = "datamart/silver/attributes/"
    silver_financials_directory = "datamart/silver/financials/"
    silver_clickstream_directory = "datamart/silver/clickstream/"
    gold_feature_store_directory = "datamart/gold/feature_store/"

    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)

    # run data processing: features for loans applied in this month (mob == 0)
    utils.data_processing_gold_table.process_gold_feature_store(
        snapshotdate,
        silver_loan_directory,
        silver_attributes_directory,
        silver_financials_directory,
        silver_clickstream_directory,
        gold_feature_store_directory,
        spark,
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
