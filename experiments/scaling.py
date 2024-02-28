from libraries import PDLibrary, DSLibrary, PolarsLibrary
from data import get_diabetes
import gc
import time
import os
import pandas as pd
import numpy as np
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter


rng = np.random.default_rng(42)


def measure_time(function):
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
    t1 = time.perf_counter(), time.process_time()

    function()

    t2 = time.perf_counter(), time.process_time()
    meter.stop()
    sample = meter.get_trace()._samples[0]
    cpu_energy = sample.energy["package_0"]
    gpu_energy = sample.energy["nvidia_gpu_0"]
    counter_diff = t2[0] - t1[0]
    cpu_time = t2[1] - t1[1]
    return cpu_energy, gpu_energy, counter_diff, cpu_time, sample.duration


def bootstrap_data(df: pd.DataFrame, sample_size: int = 10_000) -> pd.DataFrame:
    """Bootstrap data in a pandas dataframe."""
    N = len(df.index)
    idx = rng.integers(low=0, high=N, size=sample_size)
    return df.iloc[idx]


def run_tests(
    dataset: pd.DataFrame,
    tests: list[str],
    library: DSLibrary,
    groupby_column: str = None,
    sort_column: str = None,
    n_repeats: int = 10,
    sample_size: int = 100_000,
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
    res = np.zeros((n_repeats, 3 * len(tests)))
    for ti in range(n_repeats):
        print(f"Start test {ti+1}/{n_repeats}")
        sdf = bootstrap_data(dataset, sample_size=sample_size)
        sdf = library.convert_from_pandas(df=sdf)
        for tj, test in enumerate(tests):
            match test:
                case "groupby":
                    cpu_energy, gpu_energy, counter, cpu_time, duration = measure_time(
                        lambda: library.groupby(sdf, groupby_column)
                    )
                case "sort":
                    cpu_energy, gpu_energy, counter, cpu_time, duration = measure_time(
                        lambda: library.sort_column(sdf, sort_column)
                    )
                case "drop_duplicates":
                    cpu_energy, gpu_energy, counter, cpu_time, duration = measure_time(lambda: library.drop_duplicates(sdf))
            res[ti, 3 * tj + 0] = cpu_energy
            res[ti, 3 * tj + 1] = counter
            res[ti, 3 * tj + 2] = duration
    res_df = pd.DataFrame(res)
    res_df.columns = [prefix + "_" + library.method_name + "_" + postfix for postfix in tests for prefix in ["µJ", "cntr", "cpu"]]
    return res_df


if __name__ == "__main__":
    # pyRAPL.setup()
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
