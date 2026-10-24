from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta

# model is trained once, at this date (labels up to here cover applications until 2023-12)
MODEL_TRAIN_DATE = "2024-06-01"
# model deployed for inference from MODEL_TRAIN_DATE onwards
MODEL_NAME = "credit_model_2024_06_01.pkl"


def is_model_train_date(ds, **kwargs):
    # True -> downstream training runs; False -> training is skipped
    return ds == MODEL_TRAIN_DATE


def is_model_deployed(ds, **kwargs):
    # True -> inference runs; False (months before the model exists) -> inference is skipped
    return ds >= MODEL_TRAIN_DATE


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 11, 1),  # last LMS snapshot (same range as assignment 1)
    catchup=True,
    max_active_runs=1,  # run months one by one in order (clickstream rolling mean needs earlier months)
) as dag:

    # --- bronze layer (4 source tables) ---
    start = EmptyOperator(task_id="start")

    bronze_loan = BashOperator(
        task_id="bronze_loan",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/bronze_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "lms_loan_daily"'
        ),
    )

    bronze_clickstream = BashOperator(
        task_id="bronze_clickstream",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/bronze_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "feature_clickstream"'
        ),
    )

    bronze_attributes = BashOperator(
        task_id="bronze_attributes",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/bronze_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "features_attributes"'
        ),
    )

    bronze_financials = BashOperator(
        task_id="bronze_financials",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/bronze_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "features_financials"'
        ),
    )

    # the 4 bronze tasks are independent of each other
    start >> [bronze_loan, bronze_clickstream, bronze_attributes, bronze_financials]


    # --- silver layer (each silver task needs its own bronze task) ---
    silver_loan = BashOperator(
        task_id="silver_loan",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/silver_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "lms_loan_daily"'
        ),
    )

    silver_clickstream = BashOperator(
        task_id="silver_clickstream",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/silver_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "feature_clickstream"'
        ),
    )

    silver_attributes = BashOperator(
        task_id="silver_attributes",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/silver_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "features_attributes"'
        ),
    )

    silver_financials = BashOperator(
        task_id="silver_financials",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/silver_table.py '
            '--snapshotdate "{{ ds }}" '
            '--tablename "features_financials"'
        ),
    )

    bronze_loan >> silver_loan
    bronze_clickstream >> silver_clickstream
    bronze_attributes >> silver_attributes
    bronze_financials >> silver_financials


    # --- gold layer ---
    # label store only needs the loan table
    gold_label_store = BashOperator(
        task_id="gold_label_store",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/gold_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # feature store joins all 4 silver tables
    gold_feature_store = BashOperator(
        task_id="gold_feature_store",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/gold_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_loan >> gold_label_store
    [silver_loan, silver_clickstream, silver_attributes, silver_financials] >> gold_feature_store


    # --- model training (only at MODEL_TRAIN_DATE) ---
    check_model_train_date = ShortCircuitOperator(
        task_id="check_model_train_date",
        python_callable=is_model_train_date,
        ignore_downstream_trigger_rules=False,  # only skip model_train, not inference further downstream
    )

    model_train = BashOperator(
        task_id="model_train",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/model_train.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    [gold_label_store, gold_feature_store] >> check_model_train_date >> model_train


    # --- model inference (every month from MODEL_TRAIN_DATE) ---
    check_model_deployed = ShortCircuitOperator(
        task_id="check_model_deployed",
        python_callable=is_model_deployed,
    )

    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            f'--modelname "{MODEL_NAME}"'
        ),
        # model_train is skipped in most months: still run if nothing failed
        trigger_rule="none_failed_min_one_success",
    )

    gold_feature_store >> check_model_deployed >> model_inference
    model_train >> model_inference  # in the training month, wait for the new model


    # --- model monitoring (every month from MODEL_TRAIN_DATE) ---
    # stability of this month's predictions + performance of predictions whose labels arrive this month
    model_monitoring = BashOperator(
        task_id="model_monitoring",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/model_monitoring.py '
            '--snapshotdate "{{ ds }}" '
            f'--modelname "{MODEL_NAME}"'
        ),
    )

    [model_inference, gold_label_store] >> model_monitoring


    # --- model visualisation: monitoring charts (PNG) refreshed every month ---
    model_visualisation = BashOperator(
        task_id="model_visualisation",
        bash_command=(
            'cd /opt/airflow/project && '
            'python3 scripts/model_visualisation.py '
            '--snapshotdate "{{ ds }}" '
            f'--modelname "{MODEL_NAME}"'
        ),
    )

    model_monitoring >> model_visualisation
