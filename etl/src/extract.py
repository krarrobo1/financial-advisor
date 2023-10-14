from src.services.storage import s3
from multiprocessing import Process, Event
from multiprocessing.connection import Connection
from typing import List
import logging
import os


def extract_docs(
    bucket_name: str, s3_folder: str, local_dir: str, directories: List[str]
):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=s3_folder)

    if not directories:
        logging.info("No directories passed!")
        return

    for directory in directories:
        directory_objects = objects.filter(Prefix=f"{s3_folder}/{directory}")
        for obj in directory_objects:
            try:
                target = (
                    obj.key
                    if local_dir is None
                    else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
                )
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                if obj.key[-1] == "/":
                    continue

                # Check if file already exists in the local directory
                if not os.path.exists(target):
                    logging.info(f"Downloading {obj.key} to {target}")
                    bucket.download_file(obj.key, target)
                    filesplit = obj.key.split("/")
                    yield {
                        "path": target,
                        "company": filesplit[1],
                        "year": filesplit[2].split("_")[2][:4],
                        "filename": filesplit[2],
                    }
                    logging.info(f"Extracted doc with path: {target}")
                else:
                    logging.info(f"Skipping {obj.key}, already exists at {target}")
            except Exception as e:
                logging.error(f"Failed to extract {obj.key}: {str(e)}")


class Extractor(Process):
    def __init__(
        self,
        connection: Connection,
        bucket_name: str,
        s3_folder: str,
        local_dir: str = None,
        close_event: Event = None,
        directories: List[str] = None,
    ):
        self.connection = connection
        self.bucket_name = bucket_name
        self.s3_folder = s3_folder
        self.local_dir = local_dir
        self.close_event = close_event
        self.directories = directories
        super().__init__()

    def run(self):
        try:
            for doc in extract_docs(
                self.bucket_name, self.s3_folder, self.local_dir, self.directories
            ):
                self.connection.send(doc)
                logging.info(f'Sent doc with path: {doc["path"]}')
        except Exception as e:
            logging.error(f"Extractor failed: {str(e)}")
        finally:
            if self.close_event:
                self.close_event.set()  # Signal the event to close the pipe
            self.connection.close()
            logging.info("Extractor process terminated.")
