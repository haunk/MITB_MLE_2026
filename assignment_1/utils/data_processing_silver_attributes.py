import os
import pandas as pd
from datetime import datetime
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, when, regexp_replace, trim
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Silver layer for attributes table.
    Cleaning rules (from EDA 1):
        - Drop PII columns: Name, SSN
        - Age: strip trailing underscores, cast to int, set values outside 18-100 to null
        - Occupation: set placeholder '_______' to null
        - Enforce schema types

    Args:
        snapshot_date_str: date string in 'YYYY-MM-DD' format
        bronze_attributes_directory: path to bronze attributes partitions
        silver_attributes_directory: path to silver output directory
        spark: SparkSession

    Returns:
        Spark DataFrame of cleaned attributes data
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load from bronze partition
    partition_name = f"bronze_features_attributes_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_attributes_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"loaded from: {filepath} | row count: {df.count()}")

    # --- clean data ---

    # 1. drop PII columns
    df = df.drop("Name", "SSN")

    # 2. clean Age: strip trailing underscores, cast to int, remove impossible values
    df = df.withColumn("Age", regexp_replace(col("Age").cast(StringType()), "_+$", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", when((col("Age") >= 18) & (col("Age") <= 100), col("Age")).otherwise(None))

    # 3. clean Occupation: replace placeholder with null
    df = df.withColumn("Occupation", when(col("Occupation") == "_______", None).otherwise(col("Occupation")))

    # --- enforce schema ---
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table as parquet
    partition_name = f"silver_features_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_attributes_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f"saved to: {filepath}")

    return df