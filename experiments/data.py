import openml
from pathlib import Path
from typing import Union
import pandas as pd


def find_path(dir_name: str = "data") -> Path:
    """Convert string to Path and create necessary parent directories."""
    path = Path(dir_name)
    if not path.is_absolute():
        # Use the path of this file to find the root directory of the project
        path = Path(__file__).parent.parent / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_dataset(id: int, cache_dir: Union[str, Path]) -> openml.OpenMLDataset:
    """Convenience function to fetch datasets from OpenML API.
    Parameters
    ----------
    id : int
        OpenML dataset id.
    cache_dir : pathlib.Path or string
        Path to local cache directory.

    Returns
    -------
    OpenML dataset object.
    """
    with open(Path(__file__).parent / "openml_apikey.txt", "r") as apifile:
        apikey = apifile.read()
    openml.config.apikey = apikey
    # OpenML can handle the caching of datasets
    openml.config.cache_directory = cache_dir
    return openml.datasets.get_dataset(
        id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )


def get_diabetes(data_dir: str = "data") -> pd.DataFrame:
    """Get Diabetes dataset."""
    dir = find_path(data_dir)
    dataset = _get_dataset(42608, dir)
    df, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    return df


def get_spam(data_dir: str = "data") -> pd.DataFrame:
    """Get spam dataset."""
    dir = find_path(data_dir)
    dataset = _get_dataset(44, dir)
    df, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    return df


if __name__ == "__main__":
    print("Downloading all datasets to cache...")
    get_diabetes()
    get_spam()
