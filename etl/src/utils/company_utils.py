import pandas as pd
from src.utils.str_utils import slugify
from src.utils.storage_utils import list_folders
from src.settings import NASDAQ_TOP_100_FILE


def get_top_100_nasdaq() -> pd.DataFrame:
    df = pd.read_excel(f"{NASDAQ_TOP_100_FILE}")
    df["Name"] = df["Description"].apply(slugify)
    return df[["Description", "Name"]]


def get_top_company_list():
    top_100_nasdaq = get_top_100_nasdaq()
    folder_names = list_folders()
    return top_100_nasdaq[top_100_nasdaq["Name"].isin(folder_names)]["Name"].to_list()
