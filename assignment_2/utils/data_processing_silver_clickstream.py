import os
from datetime import datetime
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DateType


def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Silver layer for clickstream table.
    From EDA 1: all 20 features are integer, no nulls, no dirty values,
    no outliers beyond expected range. No cleaning needed.

    This layer enforces schema types and converts from CSV to parquet
    for consistent format across the silver layer.

    Args:
        snapshot_date_str: date string in 'YYYY-MM-DD' format
        bronze_clickstream_directory: path to bronze clickstream partitions
        silver_clickstream_directory: path to silver output directory
        spark: SparkSession

    Returns:
        Spark DataFrame of clickstream data
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load from bronze partition
    partition_name = f"bronze_feature_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_clickstream_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"loaded from: {filepath} | row count: {df.count()}")

    # --- enforce schema ---
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    for c in fe_cols:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table as parquet
    partition_name = f"silver_feature_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_clickstream_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f"saved to: {filepath}")

    return df