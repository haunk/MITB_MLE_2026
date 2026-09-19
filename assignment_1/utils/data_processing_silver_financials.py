import os
import pandas as pd
from datetime import datetime
import re
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, when, regexp_replace, trim, lit, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Silver layer for financials table.
    Cleaning rules (from EDA 1, section 3.5):

    Trailing underscores:
        - Annual_Income, Num_of_Loan, Num_of_Delayed_Payment,
          Changed_Credit_Limit, Outstanding_Debt,
          Amount_invested_monthly, Monthly_Balance: strip underscores

    Placeholder/junk to null:
        - Credit_Mix: '_' to null
        - Payment_Behaviour: '!@9#%8' to null
        - Amount_invested_monthly: 10000 to null (placeholder __10000__)
        - Payment_of_Min_Amount: 'NM' to null

    Impossible values to null:
        - Num_Bank_Accounts: negatives or >20
        - Num_Credit_Card: >20
        - Interest_Rate: >50
        - Num_of_Loan: negatives or >10
        - Num_of_Delayed_Payment: negatives
        - Num_Credit_Inquiries: >20
        - Total_EMI_per_month: >600 (above 95th percentile)
        - Monthly_Balance: abs value > 10000 (catches garbage like -3.33e+26)

    Feature engineering:
        - Credit_History_Age: parse 'X Years and Y Months' to total months
        - Payment_Behaviour: split into two ordinal columns
            spending_level (Low=0, High=1)
            payment_value_level (Small=0, Medium=1, Large=2)

    Drop columns:
        - Type_of_Loan (too many unique combos, high missing rate)
        - Payment_Behaviour (replaced by spending_level + payment_value_level)

    Args:
        snapshot_date_str: date string in 'YYYY-MM-DD' format
        bronze_financials_directory: path to bronze financials partitions
        silver_financials_directory: path to silver output directory
        spark: SparkSession

    Returns:
        Spark DataFrame of cleaned financials data
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load from bronze partition
    partition_name = f"bronze_features_financials_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_financials_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"loaded from: {filepath} | row count: {df.count()}")

    # --- drop columns ---
    df = df.drop("Type_of_Loan")

    # --- strip trailing underscores from numeric string columns ---
    underscore_cols = [
        "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt",
        "Amount_invested_monthly", "Monthly_Balance",
    ]
    for c in underscore_cols:
        df = df.withColumn(c, regexp_replace(col(c).cast(StringType()), r"^_+|_+$", ""))

    # --- placeholder / junk values to null ---

    # Credit_Mix: standalone '_' to null
    df = df.withColumn(
        "Credit_Mix",
        when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")),
    )

    # Payment_Behaviour: '!@9#%8' to null, then encode into two ordinal columns
    # Format: '{Spending}_spent_{Value}_value_payments'
    # spending_level: Low=0, High=1
    # payment_value_level: Small=0, Medium=1, Large=2
    df = df.withColumn(
        "spending_level",
        when(col("Payment_Behaviour").contains("Low_spent"), 0)
        .when(col("Payment_Behaviour").contains("High_spent"), 1)
        .otherwise(None)
        .cast(IntegerType()),
    )
    df = df.withColumn(
        "payment_value_level",
        when(col("Payment_Behaviour").contains("Small_value"), 0)
        .when(col("Payment_Behaviour").contains("Medium_value"), 1)
        .when(col("Payment_Behaviour").contains("Large_value"), 2)
        .otherwise(None)
        .cast(IntegerType()),
    )
    df = df.drop("Payment_Behaviour")

    # Payment_of_Min_Amount: 'NM' to null
    df = df.withColumn(
        "Payment_of_Min_Amount",
        when(col("Payment_of_Min_Amount") == "NM", None).otherwise(col("Payment_of_Min_Amount")),
    )

    # --- cast numeric columns ---
    df = df.withColumn("Annual_Income", col("Annual_Income").cast(FloatType()))
    df = df.withColumn("Num_of_Loan", col("Num_of_Loan").cast(IntegerType()))
    df = df.withColumn("Num_of_Delayed_Payment", col("Num_of_Delayed_Payment").cast(IntegerType()))
    df = df.withColumn("Changed_Credit_Limit", col("Changed_Credit_Limit").cast(FloatType()))
    df = df.withColumn("Outstanding_Debt", col("Outstanding_Debt").cast(FloatType()))
    df = df.withColumn("Amount_invested_monthly", col("Amount_invested_monthly").cast(FloatType()))
    df = df.withColumn("Monthly_Balance", col("Monthly_Balance").cast(FloatType()))

    # --- impossible values to null ---

    # Num_Bank_Accounts: negatives or >20
    df = df.withColumn(
        "Num_Bank_Accounts",
        when((col("Num_Bank_Accounts") >= 0) & (col("Num_Bank_Accounts") <= 20), col("Num_Bank_Accounts")),
    )

    # Num_Credit_Card: >20
    df = df.withColumn(
        "Num_Credit_Card",
        when((col("Num_Credit_Card") >= 0) & (col("Num_Credit_Card") <= 20), col("Num_Credit_Card")),
    )

    # Interest_Rate: >50
    df = df.withColumn(
        "Interest_Rate",
        when((col("Interest_Rate") >= 0) & (col("Interest_Rate") <= 50), col("Interest_Rate")),
    )

    # Num_of_Loan: negatives or >10
    df = df.withColumn(
        "Num_of_Loan",
        when((col("Num_of_Loan") >= 0) & (col("Num_of_Loan") <= 10), col("Num_of_Loan")),
    )

    # Num_of_Delayed_Payment: negatives
    df = df.withColumn(
        "Num_of_Delayed_Payment",
        when(col("Num_of_Delayed_Payment") >= 0, col("Num_of_Delayed_Payment")),
    )

    # Num_Credit_Inquiries: >20
    df = df.withColumn(
        "Num_Credit_Inquiries",
        when((col("Num_Credit_Inquiries") >= 0) & (col("Num_Credit_Inquiries") <= 20), col("Num_Credit_Inquiries")),
    )

    # Total_EMI_per_month: above 95th percentile (~600)
    df = df.withColumn(
        "Total_EMI_per_month",
        when(col("Total_EMI_per_month") <= 600, col("Total_EMI_per_month")),
    )

    # Amount_invested_monthly: 10000 is placeholder
    df = df.withColumn(
        "Amount_invested_monthly",
        when(col("Amount_invested_monthly") < 10000, col("Amount_invested_monthly")),
    )

    # Monthly_Balance: extreme values (garbage like -3.33e+26)
    df = df.withColumn(
        "Monthly_Balance",
        when(
            (col("Monthly_Balance") > -10000) & (col("Monthly_Balance") < 10000),
            col("Monthly_Balance"),
        ),
    )

    # --- feature engineering ---

    # Credit_History_Age: parse 'X Years and Y Months' to total months
    df = df.withColumn(
        "Credit_History_Age",
        (
            F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years?", 1).cast(IntegerType()) * 12
            + F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months?", 1).cast(IntegerType())
        ),
    )

    # --- enforce final schema ---
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": IntegerType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "spending_level": IntegerType(),
        "payment_value_level": IntegerType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table as parquet
    partition_name = f"silver_features_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_financials_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f"saved to: {filepath}")

    return df