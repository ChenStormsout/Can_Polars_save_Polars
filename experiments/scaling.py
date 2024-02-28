from libraries import PDLibrary, DSLibrary, PolarsLibrary
from data import get_diabetes
from typing import Callable
import gc
import time
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)


def measure_time(function) -> float:
    """Measure CPU and wall times of a given function."""
    gc.collect()
    t1 = time.process_time()
    function()
    t2 = time.process_time()
    return t2 - t1


def bootstrap_data(df: pd.DataFrame, sample_size: int = 10_000) -> pd.DataFrame:
    """Bootstrap data in a pandas dataframe."""
    N = len(df.index)
    idx = rng.integers(low=0, high=N, size=sample_size)
    return df.iloc[idx]


def run_tests(
    dataset: pd.DataFrame,
    tests: list[str],
    library: DSLibrary,
    metric_function: Callable[[Callable], float] = measure_time,
    groupby_column: str = None,
    sort_column: str = None,
    n_repeats: int = 10,
    sample_sizes: list[int] = [100_000],
) -> pd.DataFrame:
    """Wrapper function to run whatever tests we want to perform.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input data.
    tests : list of strings
        List of test names to be performed.
    library : DSLibrary instance
        Library to perform the tests with.
    metric_function : Callable
        Function wrapper which wraps the computation, measures some quantity of interest
        and returns the metric value.
    groupby_column : string, optional
        Column name over which the groupby test is performed. If left empty, the test is
        skipped.
    sort_column : string, optional
        Column name over which the sort test is performed. If left empty, the test is
        skipped.
    n_repeats : int, optional.
        How many times tests are repeated (with different samples). Defaults to 10.
    sample_size : int, optional.
        Bootstrap sample size. Defaults to 100 000.
    """
    res = np.zeros(shape=(len(sample_sizes), n_repeats, len(tests)))
    for si, sample_size in enumerate(sample_sizes):
        print(
            f"Start tests with sample size {sample_size} ({si+1}/{len(sample_sizes)})"
        )
        for ti in range(n_repeats):
            print(f"Start test {ti+1}/{n_repeats}")
            sdf = bootstrap_data(dataset, sample_size=sample_size)
            sdf = library.convert_from_pandas(df=sdf)
            for tj, test in enumerate(tests):
                match test:
                    case "groupby":
                        cpu_time = metric_function(
                            lambda: library.groupby(sdf, groupby_column)
                        )
                    case "sort":
                        cpu_time = metric_function(
                            lambda: library.sort_column(sdf, sort_column)
                        )
                    case "drop_duplicates":
                        cpu_time = metric_function(lambda: library.drop_duplicates(sdf))
                res[si, ti, tj] = cpu_time
    res_dfs = []
    for si, sample_size in enumerate(sample_sizes):
        pdf = pd.DataFrame(res[si, :, :])
        pdf["n"] = sample_size
        res_dfs.append(pdf)
    res_df = pd.concat(res_dfs)
    res_df.columns = [library.method_name + "_" + t for t in tests] + ["n"]
    return res_df


if __name__ == "__main__":
    df = get_diabetes()
    res_pd = run_tests(
        dataset=df,
        tests=["drop_duplicates", "groupby", "sort"],
        library=PDLibrary(),
        groupby_column="Pregnancies",
        sort_column="Glucose",
    )
    print(res_pd)
    res_pl = run_tests(
        dataset=df,
        tests=["drop_duplicates", "groupby", "sort"],
        library=PolarsLibrary(),
        groupby_column="Pregnancies",
        sort_column="Glucose",
    )
    print(res_pl)
