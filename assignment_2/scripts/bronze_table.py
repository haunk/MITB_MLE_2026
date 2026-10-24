import argparse
import os
import pyspark

import utils.data_processing_bronze_table

# to call this script: python scripts/bronze_table.py --snapshotdate "2023-01-01" --tablename "lms_loan_daily"

# source table name -> bronze output directory
BRONZE_DIRECTORIES = {
    "lms_loan_daily": "datamart/bronze/loan/",
    "feature_clickstream": "datamart/bronze/clickstream/",
    "features_attributes": "datamart/bronze/attributes/",
    "features_financials": "datamart/bronze/financials/",
}


def main(snapshotdate, tablename):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # create bronze datalake
    bronze_directory = BRONZE_DIRECTORIES[tablename]

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    # run data processing
    utils.data_processing_bronze_table.process_bronze_table(
        snapshotdate, bronze_directory, spark, data_file_name=tablename
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--tablename", type=str, required=True, choices=list(BRONZE_DIRECTORIES.keys()), help="source table name")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.tablename)
