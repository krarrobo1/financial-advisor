import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
FOLDER_NAME = os.getenv("FOLDER_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
LOCAL_DIR = os.getenv("LOCAL_DIR", "/opt/airflow/dags/dataset")
NASDAQ_TOP_100_FILE = f"{LOCAL_DIR}/top_100_nasdaq_technology.xlsx"
REDIS_PORT = 6379
REDIS_DB_ID = 0
REDIS_IP = os.getenv("REDIS_IP", "redis")
PROCESSED_KEY = "processed_paths"
