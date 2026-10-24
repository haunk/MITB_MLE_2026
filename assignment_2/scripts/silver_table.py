import argparse
import os
import pyspark

import utils.data_processing_silver_loan
import utils.data_processing_silver_clickstream
import utils.data_processing_silver_attributes
import utils.data_processing_silver_financials

# to call this script: python scripts/silver_table.py --snapshotdate "2023-01-01" --tablename "lms_loan_daily"

# source table name -> (silver processing function, bronze input directory, silver output directory)
SILVER_JOBS = {
    "lms_loan_daily": (
        utils.data_processing_silver_loan.process_silver_loan,
        "datamart/bronze/loan/",
        "datamart/silver/loan/",
    ),
    "feature_clickstream": (
        utils.data_processing_silver_clickstream.process_silver_clickstream,
        "datamart/bronze/clickstream/",
        "datamart/silver/clickstream/",
    ),
    "features_attributes": (
        utils.data_processing_silver_attributes.process_silver_attributes,
        "datamart/bronze/attributes/",
        "datamart/silver/attributes/",
    ),
    "features_financials": (
        utils.data_processing_silver_financials.process_silver_financials,
        "datamart/bronze/financials/",
        "datamart/silver/financials/",
    ),
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

    # create silver datalake
    process_silver_function, bronze_directory, silver_directory = SILVER_JOBS[tablename]

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    # run data processing
    process_silver_function(snapshotdate, bronze_directory, silver_directory, spark)

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--tablename", type=str, required=True, choices=list(SILVER_JOBS.keys()), help="source table name")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.tablename)
