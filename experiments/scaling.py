from libraries import PDLibrary, DSLibrary, PolarsLibrary, CuDFLibrary
from data import get_diabetes
from load_dataset import load_parking_data_pandas
from typing import Callable, Any
from dataclasses import dataclass
import gc
import time
import pandas as pd
import polars as pl
import numpy as np
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter

rng = np.random.default_rng(42)

@dataclass
class Measure:
    duration_seconds: float
    cpu_µJ: float
    gpu_µJ: float

def measure_time(function) -> float:
    """Measure CPU and wall times of a given function."""
    gc.collect()
    domains = [
        RaplPackageDomain(0),
        # RaplDramDomain(0),
        NvidiaGPUDomain(0),
    ]
    devices = DeviceFactory.create_devices(domains)
    meter = EnergyMeter(devices)
    meter.start()

    function()

    meter.stop()
    sample = meter.get_trace()._samples[0]
    cpu_energy = sample.energy["package_0"]
    gpu_energy = sample.energy["nvidia_gpu_0"]
    duration = sample.duration
    return Measure(duration, cpu_energy, gpu_energy)


def bootstrap_data(df: pd.DataFrame, sample_size: int = 10_000) -> pd.DataFrame:
    """Bootstrap data in a pandas dataframe."""
    N = df.shape[0]
    idx = rng.integers(low=0, high=N, size=sample_size)
    return df.iloc[idx]


def run_tests(
    dataset: pd.DataFrame,
    tests: list[str],
    library: DSLibrary,
    metric_function: Callable[[Callable], Measure] = measure_time,
    groupby_column: str = None,
    sort_column: str = None,
    merge_column: str = None,
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
    res = np.zeros(shape=(len(sample_sizes), n_repeats, len(tests) * 3))
    for si, sample_size in enumerate(sample_sizes):
        print(
            f"Start tests with sample size {sample_size} ({si+1}/{len(sample_sizes)})"
        )
        for ti in range(n_repeats):
            print(f"Start test {ti+1}/{n_repeats}")
            bdf = bootstrap_data(df, sample_size=sample_size)
            small = bootstrap_data(df, sample_size=100)
            sdf = library.convert_from_pandas(df=bdf)
            for tj, test in enumerate(tests):
                match test:
                    case "load":
                        if si == 0:
                            metric = metric_function(
                                lambda: library.load()
                            )
                        else:
                            metric = Measure(0,0,0)
                    case "groupby":
                        metric = metric_function(
                            lambda: library.groupby(sdf, groupby_column)
                        )
                    case "sort":
                        metric = metric_function(
                            lambda: library.sort_column(sdf, sort_column)
                        )
                    case "drop_duplicates":
                        metric = metric_function(lambda: library.drop_duplicates(sdf))
                    case "merge":
                        metric = metric_function(
                            lambda: library.merge(
                                sdf, df_b=small, merge_column=merge_column
                            )
                        )
                res[si, ti, 3 * tj + 0] = metric.duration_seconds
                res[si, ti, 3 * tj + 1] = metric.cpu_µJ
                res[si, ti, 3 * tj + 2] = metric.gpu_µJ
    res_dfs = []
    for si, sample_size in enumerate(sample_sizes):
        pdf = pd.DataFrame(res[si, :, :])
        pdf["n"] = sample_size
        res_dfs.append(pdf)
    res_df = pd.concat(res_dfs)
    res_df.columns = [prefix + "_" + library.method_name + "_" + t for t in tests for prefix in ["secs", "cpu_µJ", "gpu_µJ"]] + ["n"]
    return res_df


if __name__ == "__main__":
    df = load_parking_data_pandas()
    print(f"Loaded {len(df)} rows")

    tests = ["load", "drop_duplicates", "groupby", "sort"]# "merge"
    sample_sizes = [100_000, 100_000_0, 100_000_00]
    res_pd = run_tests(
        dataset=df,
        tests=tests,
        library=PDLibrary(),
        groupby_column="Street Code1",
        sort_column="Issue Date",
        merge_column="Street Code1",
        sample_sizes=sample_sizes,
    )
    print(res_pd)
    res_pd.to_csv("pandas.csv")
    res_pl = run_tests(
        dataset=df,
        tests=tests,
        library=PolarsLibrary(),
        groupby_column="Street Code1",
        sort_column="Issue Date",
        merge_column="Street Code1",
        sample_sizes=sample_sizes,
    )
    print(res_pl)
    res_pl.to_csv("polars.csv")
    res_cd = run_tests(
        dataset=df,
        tests=tests,
        library=CuDFLibrary(),
        groupby_column="Street Code1",
        sort_column="Issue Date",
        merge_column="Street Code1",
        sample_sizes=sample_sizes,
    )
    print(res_cd)
    res_cd.to_csv("cudf.csv")
