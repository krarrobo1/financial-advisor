__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from airflow.decorators import dag, task
from src.jobs import extract_documents, load_documents_recursively
from src.settings import LOCAL_DIR
import pendulum
import logging


@dag(
    schedule=None,
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def elt_taskflow():
    """
    This dag is in charge of ETL
    """

    @task()
    def extract_task():
        extract_documents()

    @task()
    def transform_and_load_task():
        logging.info("Starting load task...")
        result = load_documents_recursively(LOCAL_DIR)
        logging.info("Finishing load task...")
        logging.info(result)

    extract_task() >> transform_and_load_task()


elt_taskflow()
