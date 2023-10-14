import logging
from jobs import extract_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="extract-job.log",
    filemode="w",
)


def main():
    extract_documents()


if __name__ == "__main__":
    main()
