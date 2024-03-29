import os
from pathlib import Path

import pandas as pd
import cudf
import polars as pl


def load_parking_data_pandas() -> pd.DataFrame:
    """Util function that loads the NYC parking data into
    a pandas DataFrame from 'nyc_parking_violations_2022.parquet'.
    If the file does not exist, download_data.sh is executed to download
    the file.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame containing the NYC parking data from
        'https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2022/7mxj-7a6y/about_data'
    """
    file_path = Path(__file__).parent.parent / "nyc_parking_violations_2022.parquet"
    if not file_path.exists():
        os.system("sh ./download_data.sh")
    return pd.read_parquet(file_path)


def load_parking_data_cudf() -> cudf.DataFrame:
    """Util function that loads the NYC parking data into
    a pandas DataFrame from 'nyc_parking_violations_2022.parquet'.
    If the file does not exist, download_data.sh is executed to download
    the file.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame containing the NYC parking data from
        'https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2022/7mxj-7a6y/about_data'
    """
    file_path = Path(__file__).parent.parent / "nyc_parking_violations_2022.parquet"
    if not file_path.exists():
        os.system("sh ./download_data.sh")
    return cudf.read_parquet(file_path)


def load_parking_data_polars() -> pl.DataFrame:
    """Util function that loads the NYC parking data into
    a pandas DataFrame from 'nyc_parking_violations_2022.parquet'.
    If the file does not exist, download_data.sh is executed to download
    the file.

    Returns
    -------
    pl.DataFrame
        pandas DataFrame containing the NYC parking data from
        'https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2022/7mxj-7a6y/about_data'
    """
    file_path = Path(__file__).parent.parent / "nyc_parking_violations_2022.parquet"
    if not file_path.exists():
        os.system("sh ./download_data.sh")
    return pl.read_parquet(file_path)
