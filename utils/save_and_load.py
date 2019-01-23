import pandas as pd
from pathlib import Path

PROJECT_PATH = Path.cwd().parent

import os

def save_obj(save_object: pd.DataFrame, file_name: str):
    """
    :param save_object: object to be saved
    :param file_name: file name of the object to be saved in ./data directory, makes directory if not exists
    :return:
    """
    dir_name = Path(PROJECT_PATH, "data")
    dir_name.mkdir(exist_ok=True, parents=True)

    save_object.to_pickle(Path(dir_name, f"{file_name}.pkl"))


def load_obj(file_name: str)-> pd.DataFrame:
    """
    :param file_name: name of pickle object to load from the ./data directory
    :return:
    """
    dir_name = Path(PROJECT_PATH, "data")

    if not dir_name.is_dir():
        raise FileNotFoundError

    return pd.read_pickle(Path(dir_name, f"{file_name}.pkl"))

