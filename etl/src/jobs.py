from src.extract import extract_docs
from src.settings import FOLDER_NAME, BUCKET_NAME, LOCAL_DIR
from src.utils.company_utils import get_top_company_list
from src.utils.load_utils import load_documents, get_vectorizer, save_db
from src.services.redis import save_processed_path_to_redis, check_path_exists_in_redis
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="extract-job.log",
    filemode="w",
)


def extract_documents():
    directories = get_top_company_list()

    bucket_name = BUCKET_NAME
    s3_folder = FOLDER_NAME
    local_dir = LOCAL_DIR

    for doc in extract_docs(bucket_name, s3_folder, local_dir, directories):
        # Process the extracted document here
        logging.info(f'Received doc with path: {doc["path"]}')


def load_documents_recursively(parent_directory: str):
    embedding_function = get_vectorizer()
    if not os.path.exists(parent_directory) or not os.path.isdir(parent_directory):
        logging.info(
            f"Parent directory '{parent_directory}' does not exist or is not a directory."
        )
        return

    # Get a list of all subdirectories in the parent directory
    subdirectories = [
        d
        for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d))
    ]

    # Iterate over the subdirectories and execute load_documents for each
    for child_directory in subdirectories[0:1]:
        full_child_directory = f"{parent_directory}/{child_directory}/"
        if check_path_exists_in_redis(full_child_directory):
            continue
        logging.info(f"Loading child directory: {full_child_directory}")
        documents = load_documents(full_child_directory)
        save_db(
            documents=documents,
            embedding_function=embedding_function,
            persistence_directory=f"{LOCAL_DIR}/chroma_db-database",
        )
        save_processed_path_to_redis(full_child_directory)
        logging.info(f"Finishing loading child directory: {full_child_directory}")
