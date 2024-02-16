import openml
from pathlib import Path
from typing import Union


def find_path(dir_name: str = "data") -> Path:
    path = Path(dir_name)
    if not path.is_absolute():
        # Use the path of this file to find the root directory of the project
        path = Path(__file__).parent.parent / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_dataset(id: int, cache_dir: Union[str, Path]) -> openml.OpenMLDataset:
    openml.config.apikey = "c3d6fbcd5a5741b5d4d4e15369a4f7fe"
    # OpenML can handle the caching of datasets
    openml.config.cache_directory = cache_dir
    return openml.datasets.get_dataset(id, download_data=True)


def get_diabetes(data_dir: str = "data"):
    dir = find_path(data_dir)
    dataset = _get_dataset(42608, dir)
    df, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    return df


if __name__ == "__main__":
    print("Downloading all datasets...")
    get_diabetes()
