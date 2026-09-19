import os
import pandas as pd
from datetime import datetime
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_directory, spark, data_file_name):
    """
    Bronze layer: raw ingestion from source CSV, partitioned by snapshot_date.
    No cleaning or type casting. Saves raw data as-is.

    Args:
        snapshot_date_str: date string in 'YYYY-MM-DD' format
        bronze_directory: path to bronze output directory (e.g. 'datamart/bronze/lms/')
        spark: SparkSession
        data_file_name: source file name without extension (e.g. 'lms_loan_daily')

    Returns:
        Spark DataFrame of ingested data for this snapshot_date
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to source back end - IRL connect to back end source system
    csv_file_path = f"data/{data_file_name}.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    df = df.filter(col("snapshot_date") == snapshot_date)

    row_count = df.count()
    print(f"{snapshot_date_str} | {data_file_name} | row count: {row_count}")

    # save bronze table to datamart - IRL connect to database to write
    # always save partition (even if empty) for auditability
    partition_name = f"bronze_{data_file_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f"saved to: {filepath}")

    return df