from libraries import PDLibrary, DSLibrary
from data import get_diabetes
import gc
import time
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)


def measure_time(function):
    gc.collect()
    t1 = time.perf_counter(), time.process_time()
    function()
    t2 = time.perf_counter(), time.process_time()
    return t2[0] - t1[0], t2[1] - t1[1]


def bootstrap_data(df: pd.DataFrame, sample_size: int = 10_000) -> pd.DataFrame:
    N = len(df.index)
    idx = rng.integers(low=0, high=N, size=sample_size)
    return df.iloc[idx]


def run_tests(
    dataset: pd.DataFrame,
    tests: list[str],
    library: DSLibrary,
    groupby_column: str,
    sort_column: str,
    n_repeats: int = 10,
    sample_size: int = 100_000,
) -> pd.DataFrame:
    res = np.zeros((n_repeats, len(tests)))
    for ti in range(n_repeats):
        print(f"Start test {ti+1}/{n_repeats}")
        sdf = bootstrap_data(dataset, sample_size=sample_size)
        sdf = library.convert_from_pandas(df=sdf)
        for tj, test in enumerate(tests):
            match test:
                case "groupby":
                    cpu_time, _ = measure_time(
                        lambda: library.groupby(sdf, groupby_column)
                    )
                case "sort":
                    cpu_time, _ = measure_time(
                        lambda: library.sort_column(sdf, sort_column)
                    )
                case "drop_duplicates":
                    cpu_time, _ = measure_time(lambda: library.drop_duplicates(sdf))
            res[ti, tj] = cpu_time
    res_df = pd.DataFrame(res)
    res_df.columns = [library.method_name + "_" + t for t in tests]
    return res_df


if __name__ == "__main__":
    df = get_diabetes()
    rdf = run_tests(
        df,
        ["drop_duplicates", "groupby", "sort"],
        PDLibrary(),
        "Pregnancies",
        "Glucose",
    )
    print(rdf)
