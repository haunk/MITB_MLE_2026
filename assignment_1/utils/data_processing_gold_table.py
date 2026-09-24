import os
import glob
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from datetime import datetime


# ====================================================================== #
# Gold Label Store
# ====================================================================== #


def process_labels_gold_table(snapshot_date_str, silver_loan_directory, gold_label_store_directory, spark, dpd, mob):

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_loan_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd) + 'dpd_' + str(mob) + 'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


# ====================================================================== #
# Gold Feature Store
# ====================================================================== #


# --- Fixed encoding maps (alphabetical for reproducibility) ---
OCCUPATION_MAP = {
    "Accountant": 0,
    "Architect": 1,
    "Developer": 2,
    "Doctor": 3,
    "Engineer": 4,
    "Entrepreneur": 5,
    "Journalist": 6,
    "Lawyer": 7,
    "Manager": 8,
    "Mechanic": 9,
    "Media_Manager": 10,
    "Musician": 11,
    "Scientist": 12,
    "Teacher": 13,
    "Writer": 14,
}

CREDIT_MIX_MAP = {
    "Bad": 0,
    "Standard": 1,
    "Good": 2,
}

PAYMENT_MIN_MAP = {
    "No": 0,
    "Yes": 1,
}


def process_gold_feature_store(
    snapshot_date_str,
    silver_loan_directory,
    silver_attributes_directory,
    silver_financials_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark,
):
    """
    Gold layer: ML-ready feature store.
    For loans originating on snapshot_date, joins customer attributes,
    financials, and rolling clickstream aggregations.
    Encodes categoricals at gold layer. Preserves NaN for tree-based models.

    Design decisions:
        1. Clickstream: rolling aggregation (mean, std, min, max) of all
           snapshots up to and including snapshot_date to prevent temporal leakage.
        2. Categorical encoding: Occupation (label), Credit_Mix (ordinal),
           Payment_of_Min_Amount (binary).
        3. NaN preserved, no imputation. Deferred to model pipeline.
        4. has_clickstream binary flag for customers without clickstream data.

    Args:
        snapshot_date_str: date string in 'YYYY-MM-DD' format
        silver_loan_directory: path to silver loan parquet directory
        silver_attributes_directory: path to silver attributes parquet directory
        silver_financials_directory: path to silver financials parquet directory
        silver_clickstream_directory: path to silver clickstream parquet directory
        gold_feature_store_directory: path to gold feature store output directory
        spark: SparkSession

    Returns:
        Spark DataFrame of ML-ready features for loans originating on snapshot_date
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_tag = snapshot_date_str.replace("-", "_")

    # ------------------------------------------------------------------ #
    # Step 1: Get new loans (mob == 0 means just originated this month)
    # ------------------------------------------------------------------ #
    loan_file = os.path.join(
        silver_loan_directory, f"silver_lms_loan_daily_{date_tag}.parquet"
    )
    df_loan = spark.read.parquet(loan_file)
    df_new_loans = df_loan.filter(col("mob") == 0).select(
        "loan_id", "Customer_ID", "loan_start_date"
    )

    new_loan_count = df_new_loans.count()
    print(f"{snapshot_date_str} | new loans: {new_loan_count}")

    # Save empty partition for auditability even when no new loans
    if new_loan_count == 0:
        empty_schema = _build_empty_schema(spark)
        partition_name = f"gold_feature_store_{date_tag}.parquet"
        filepath = os.path.join(gold_feature_store_directory, partition_name)
        empty_schema.write.mode("overwrite").parquet(filepath)
        print(f"saved to: {filepath} (empty)")
        return empty_schema

    # ------------------------------------------------------------------ #
    # Step 2: Load attributes for this snapshot_date
    # ------------------------------------------------------------------ #
    attr_file = os.path.join(
        silver_attributes_directory,
        f"silver_features_attributes_{date_tag}.parquet",
    )
    df_attr = spark.read.parquet(attr_file).drop("snapshot_date")
    df_attr = df_attr.dropDuplicates(["Customer_ID"])

    # ------------------------------------------------------------------ #
    # Step 3: Load financials for this snapshot_date
    # ------------------------------------------------------------------ #
    fin_file = os.path.join(
        silver_financials_directory,
        f"silver_features_financials_{date_tag}.parquet",
    )
    df_fin = spark.read.parquet(fin_file).drop("snapshot_date")
    df_fin = df_fin.dropDuplicates(["Customer_ID"])

    # ------------------------------------------------------------------ #
    # Step 4: Rolling clickstream aggregation (all months <= snapshot_date)
    # ------------------------------------------------------------------ #
    df_click_agg = _aggregate_clickstream(
        silver_clickstream_directory, snapshot_date, spark
    )

    # ------------------------------------------------------------------ #
    # Step 5: Join all sources on Customer_ID
    # ------------------------------------------------------------------ #
    df_features = df_new_loans.join(df_attr, on="Customer_ID", how="left")
    df_features = df_features.join(df_fin, on="Customer_ID", how="left")

    if df_click_agg is not None:
        df_features = df_features.join(df_click_agg, on="Customer_ID", how="left")

    # ------------------------------------------------------------------ #
    # Step 6: Add has_clickstream binary flag
    # ------------------------------------------------------------------ #
    if "clickstream_count" in df_features.columns:
        df_features = df_features.withColumn(
            "has_clickstream",
            when(col("clickstream_count").isNotNull(), 1).otherwise(0),
        )
        df_features = df_features.drop("clickstream_count")
    else:
        df_features = df_features.withColumn("has_clickstream", lit(0))

    # ------------------------------------------------------------------ #
    # Step 7: Encode categoricals
    # ------------------------------------------------------------------ #
    df_features = _encode_categoricals(df_features)

    # ------------------------------------------------------------------ #
    # Step 8: Drop intermediate columns, add snapshot_date, save
    # ------------------------------------------------------------------ #
    df_features = df_features.drop("loan_start_date")
    df_features = df_features.withColumn(
        "snapshot_date", lit(snapshot_date).cast("date")
    )

    row_count = df_features.count()
    partition_name = f"gold_feature_store_{date_tag}.parquet"
    filepath = os.path.join(gold_feature_store_directory, partition_name)
    df_features.write.mode("overwrite").parquet(filepath)
    print(f"saved to: {filepath} | features: {row_count} rows x {len(df_features.columns)} cols")

    return df_features


# ====================================================================== #
# Helper functions (for feature store)
# ====================================================================== #


def _aggregate_clickstream(silver_clickstream_directory, snapshot_date, spark):
    """
    Rolling aggregation of clickstream features.
    Reads all parquet files with date <= snapshot_date, then computes
    mean, std, min, max per customer for fe_1 through fe_20.
    """
    all_files = glob.glob(
        os.path.join(silver_clickstream_directory, "*.parquet")
    )

    valid_files = []
    for f in all_files:
        fname = os.path.basename(f)
        # Parse date: silver_feature_clickstream_2023_01_01.parquet
        date_part = (
            fname.replace("silver_feature_clickstream_", "")
            .replace(".parquet", "")
        )
        file_date_str = date_part.replace("_", "-")
        try:
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if file_date <= snapshot_date:
            valid_files.append(f)

    if not valid_files:
        return None

    df_click = spark.read.parquet(*valid_files)

    # Check for empty DataFrame
    if df_click.count() == 0:
        return None

    fe_cols = [f"fe_{i}" for i in range(1, 21)]

    # Build aggregation expressions: mean, std, min, max per feature
    agg_exprs = []
    for fe in fe_cols:
        agg_exprs.extend(
            [
                F.mean(fe).alias(f"{fe}_mean"),
                F.stddev(fe).alias(f"{fe}_std"),
                F.min(fe).alias(f"{fe}_min"),
                F.max(fe).alias(f"{fe}_max"),
            ]
        )
    # Count of clickstream snapshots available (useful for diagnostics)
    agg_exprs.append(F.count("*").alias("clickstream_count"))

    df_click_agg = df_click.groupBy("Customer_ID").agg(*agg_exprs)

    return df_click_agg


def _encode_categoricals(df):
    """
    Encode categorical columns at gold layer:
    - Occupation: label encoding (alphabetical integer map)
    - Credit_Mix: ordinal encoding (Bad=0, Standard=1, Good=2)
    - Payment_of_Min_Amount: binary encoding (No=0, Yes=1)

    Unknown or placeholder values ('_', '_______') mapped to None.
    NaN is preserved for all columns.
    """
    # --- Occupation ---
    if "Occupation" in df.columns:
        mapping_expr = F.create_map(
            [lit(x) for pair in OCCUPATION_MAP.items() for x in pair]
        )
        df = df.withColumn("Occupation", mapping_expr[col("Occupation")])

    # --- Credit_Mix ---
    if "Credit_Mix" in df.columns:
        mapping_expr = F.create_map(
            [lit(x) for pair in CREDIT_MIX_MAP.items() for x in pair]
        )
        df = df.withColumn("Credit_Mix", mapping_expr[col("Credit_Mix")])

    # --- Payment_of_Min_Amount ---
    if "Payment_of_Min_Amount" in df.columns:
        mapping_expr = F.create_map(
            [lit(x) for pair in PAYMENT_MIN_MAP.items() for x in pair]
        )
        df = df.withColumn(
            "Payment_of_Min_Amount",
            mapping_expr[col("Payment_of_Min_Amount")],
        )

    return df


def _build_empty_schema(spark):
    """
    Return an empty DataFrame with the full gold feature store schema.
    Used for empty partitions to maintain schema consistency.
    """
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        FloatType,
        DoubleType,
        DateType,
    )

    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    fe_agg_fields = []
    for fe in fe_cols:
        fe_agg_fields.extend(
            [
                StructField(f"{fe}_mean", DoubleType(), True),
                StructField(f"{fe}_std", DoubleType(), True),
                StructField(f"{fe}_min", DoubleType(), True),
                StructField(f"{fe}_max", DoubleType(), True),
            ]
        )

    schema = StructType(
        [
            StructField("Customer_ID", StringType(), True),
            StructField("loan_id", StringType(), True),
            # Attributes
            StructField("Age", DoubleType(), True),
            StructField("Occupation", IntegerType(), True),
            # Financials (20 numeric columns)
            StructField("Annual_Income", DoubleType(), True),
            StructField("Monthly_Inhand_Salary", DoubleType(), True),
            StructField("Num_Bank_Accounts", DoubleType(), True),
            StructField("Num_Credit_Card", DoubleType(), True),
            StructField("Interest_Rate", DoubleType(), True),
            StructField("Num_of_Loan", DoubleType(), True),
            StructField("Delay_from_due_date", IntegerType(), True),
            StructField("Num_of_Delayed_Payment", DoubleType(), True),
            StructField("Changed_Credit_Limit", DoubleType(), True),
            StructField("Num_Credit_Inquiries", DoubleType(), True),
            StructField("Credit_Mix", IntegerType(), True),
            StructField("Outstanding_Debt", DoubleType(), True),
            StructField("Credit_Utilization_Ratio", DoubleType(), True),
            StructField("Credit_History_Age", IntegerType(), True),
            StructField("Payment_of_Min_Amount", IntegerType(), True),
            StructField("Total_EMI_per_month", DoubleType(), True),
            StructField("Amount_invested_monthly", DoubleType(), True),
            StructField("Monthly_Balance", DoubleType(), True),
            StructField("spending_level", DoubleType(), True),
            StructField("payment_value_level", DoubleType(), True),
        ]
        + fe_agg_fields
        + [
            StructField("has_clickstream", IntegerType(), True),
            StructField("snapshot_date", DateType(), True),
        ]
    )

    return spark.createDataFrame([], schema)