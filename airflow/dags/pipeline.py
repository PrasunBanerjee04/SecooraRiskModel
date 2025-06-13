import os
import json
import pickle
import requests
import pandas as pd
from datetime import datetime

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'pravat',                      
    'depends_on_past': False,               
    'start_date': days_ago(1),              
    'email': ['youremail@example.com'],     
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,                          
    'retry_delay': timedelta(minutes=5),    
}

API_URL = "https://api.sealevelsensors.org/v1.0/Datastreams(262)/Observations"

DAG_FOLDER = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DAG_FOLDER, "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.pkl")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "data", "predictions.csv")

def extract_api_value(**context):
    """
    1) Compute today's UTC date (YYYY-MM-DD).
    2) Build a URL that orders by phenomenonTime desc and filters phenomenonTime
       to be between today at 00:00:00Z and today at 23:59:59Z.
    3) GET that URL, parse JSON, grab the first observation's "result".
    4) Push that numeric result into XCom under key 'api_value'.
    """
    today = datetime.utcnow().date().isoformat()
    start_date = f"{today}T00:00:00Z"
    end_date   = f"{today}T17:59:59Z"

    requested_url = (
        f"{API_URL}"
        f"?$orderby=phenomenonTime%20desc"
        f"&$filter=phenomenonTime%20ge%20{start_date}"
        f"%20and%20phenomenonTime%20le%20{end_date}"
    )

    response = requests.get(requested_url, timeout=10)
    response.raise_for_status()
    data = response.json()

    observations = data.get("value", [])
    if not observations:
        raise ValueError(f"No observations returned for {today}")

    latest_obs = observations[0]
    latest_value = latest_obs.get("result")
    if latest_value is None:
        raise KeyError(f"Couldn't find 'result' in the first observation for {today}")

    print(f"[extract_api_value] Fetched latest_value = {latest_value} (from {latest_obs.get('phenomenonTime')})")
    context['ti'].xcom_push(key='api_value', value=latest_value)


def predict_with_model(**context):
    """
    1) Pull the numeric input from XCom (from extract_api_value)
    2) Load the pickled model from disk (model/model.pkl)
    3) Run model.predict([[numeric_input]]) and push prediction into XCom
    """
    ti = context["ti"]
    numeric_input = ti.xcom_pull(key="api_value", task_ids="extract_task")

    if numeric_input is None:
        raise ValueError("No API value found in XCom for predict step.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = [[numeric_input]]
    prediction = model.predict(X)[0]

    print(f"[predict_with_model] Input = {numeric_input}, Prediction = {prediction}")
    ti.xcom_push(key="prediction", value=prediction)


def load_to_csv(**context):
    """
    1) Pull the prediction from XCom
    2) Append (timestamp, prediction) as a new row into data/predictions.csv
    """
    ti = context["ti"]
    prediction = ti.xcom_pull(key="prediction", task_ids="predict_task")

    if prediction is None:
        raise ValueError("No prediction found in XCom for load step.")

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if not os.path.exists(PREDICTIONS_CSV):
        df = pd.DataFrame([[now, prediction]], columns=["timestamp", "prediction"])
        df.to_csv(PREDICTIONS_CSV, index=False)
        print(f"[load_to_csv] Created new CSV and wrote row: {now}, {prediction}")
    else:
        df = pd.DataFrame([[now, prediction]], columns=["timestamp", "prediction"])
        df.to_csv(PREDICTIONS_CSV, mode="a", header=False, index=False)
        print(f"[load_to_csv] Appended row: {now}, {prediction}")


with DAG(
    dag_id="secoora_risk_model_pipeline",
    default_args=default_args,
    description="ETL pipeline: extract water level, predict next value, save to CSV",
    schedule_interval="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["secoora", "ml", "prediction"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_task",
        python_callable=extract_api_value,
        provide_context=True,
    )

    predict_task = PythonOperator(
        task_id="predict_task",
        python_callable=predict_with_model,
        provide_context=True,
    )

    load_task = PythonOperator(
        task_id="load_task",
        python_callable=load_to_csv,
        provide_context=True,
    )

    extract_task >> predict_task >> load_task