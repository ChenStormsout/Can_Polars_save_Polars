from abc import ABC, abstractmethod
from typing import Union, Any
from pathlib import Path
import pandas as pd


class DSLibrary(ABC):
    """Abstract base class to act as a blueprint for libraries to experiment.
    Any python library to be tested should implement at least these methods."""

    def __init__(self, method_name: str) -> None:
        self.method_name = method_name

    @abstractmethod
    def load_csv(self, filename: Union[str, Path], **kwargs) -> Any:
        """Load data from disk."""
        pass

    @abstractmethod
    def convert_from_pandas(self, df: pd.DataFrame) -> Any:
        """Convert from pandas df to whatever is the natural object for this library."""
        pass

    @abstractmethod
    def drop_duplicates(self, df: Any) -> Any:
        """Drop duplicated rows."""
        pass

    @abstractmethod
    def groupby(self, df: Any, column_name: str) -> Any:
        """Group data by column."""
        pass

    @abstractmethod
    def sort_column(self, df: Any, column_name: str) -> Any:
        """Sort dataframe by column."""
        pass

    @abstractmethod
    def merge(self, df_a: Any, df_b: Any, merge_column: str) -> Any:
        """Merge two data structures based on column."""
        pass


class PDLibrary(DSLibrary):
    """Pandas library test object."""

    def __init__(self) -> None:
        super().__init__("pandas")

    def load_csv(self, filename: str | Path, **kwargs) -> Any:
        return pd.read_csv(filename, **kwargs)

    def convert_from_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def groupby(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        return df.groupby(by=column_name)

    def sort_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        return df.sort_values(by=column_name)

    def merge(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame, merge_column: str
    ) -> pd.DataFrame:
        return df_a.merge(df_b, on=merge_column)
