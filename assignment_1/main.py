import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_loan
import utils.data_processing_silver_attributes
import utils.data_processing_silver_financials
import utils.data_processing_silver_clickstream
import utils.data_processing_gold_table

os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

print("Script started")

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2025-11-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))

        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)


# ============================================================
# Bronze Layer
# ============================================================
bronze_loan_directory = "datamart/bronze/loan/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"
bronze_attributes_directory = "datamart/bronze/attributes/"
bronze_financials_directory = "datamart/bronze/financials/"

for d in [bronze_loan_directory, bronze_clickstream_directory,
          bronze_attributes_directory, bronze_financials_directory]:
    if not os.path.exists(d):
        os.makedirs(d)

# run bronze backfill for all tables
bronze_tables = [
    ("lms_loan_daily", bronze_loan_directory),
    ("feature_clickstream", bronze_clickstream_directory),
    ("features_attributes", bronze_attributes_directory),
    ("features_financials", bronze_financials_directory),
]

for data_file_name, directory in bronze_tables:
    print(f"\n=== Processing {data_file_name} ===")
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(
            date_str, directory, spark, data_file_name=data_file_name
        )


# ============================================================
# Silver Layer
# ============================================================
silver_loan_directory = "datamart/silver/loan/"
silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"
silver_clickstream_directory = "datamart/silver/clickstream/"

for d in [silver_loan_directory, silver_attributes_directory,
          silver_financials_directory, silver_clickstream_directory]:
    if not os.path.exists(d):
        os.makedirs(d)

# run silver backfill for each table
for date_str in dates_str_lst:
    utils.data_processing_silver_loan.process_silver_loan(
        date_str, bronze_loan_directory, silver_loan_directory, spark
    )

for date_str in dates_str_lst:
    utils.data_processing_silver_attributes.process_silver_attributes(
        date_str, bronze_attributes_directory, silver_attributes_directory, spark
    )

for date_str in dates_str_lst:
    utils.data_processing_silver_financials.process_silver_financials(
        date_str, bronze_financials_directory, silver_financials_directory, spark
    )

for date_str in dates_str_lst:
    utils.data_processing_silver_clickstream.process_silver_clickstream(
        date_str, bronze_clickstream_directory, silver_clickstream_directory, spark
    )


# ============================================================
# Gold Layer
# ============================================================
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"

for d in [gold_label_store_directory, gold_feature_store_directory]:
    if not os.path.exists(d):
        os.makedirs(d)

# run gold label store backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(
        date_str,
        silver_loan_directory,
        gold_label_store_directory,
        spark,
        dpd=30,
        mob=6
    )

# run gold feature store backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_gold_feature_store(
        date_str,
        silver_loan_directory,
        silver_attributes_directory,
        silver_financials_directory,
        silver_clickstream_directory,
        gold_feature_store_directory,
        spark,
    )


# ============================================================
# Inspect Gold Output
# ============================================================

# inspect gold label store
folder_path = gold_label_store_directory
files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df_ls = spark.read.option("header", "true").parquet(*files_list)
print("label_store row_count:", df_ls.count())
df_ls.show(10)

# inspect gold feature store
folder_path = gold_feature_store_directory
files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df_fs = spark.read.option("header", "true").parquet(*files_list)
print("feature_store row_count:", df_fs.count())
df_fs.show(5)