import argparse
import os
import pyspark

import utils.data_processing_gold_table

# to call this script: python scripts/gold_label_store.py --snapshotdate "2023-01-01"


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
    gold_label_store_directory = "datamart/gold/label_store/"

    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    # run data processing: label = 1 if dpd >= 30 at month-on-book 6 (30dpd_6mob)
    utils.data_processing_gold_table.process_labels_gold_table(
        snapshotdate, silver_loan_directory, gold_label_store_directory, spark, dpd=30, mob=6
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
