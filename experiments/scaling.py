from libraries import PDLibrary, DSLibrary, PolarsLibrary
from data import get_diabetes
import gc
import time
import os
import pandas as pd
# from pypapi import papi_high
# from pypapi import events as papi_events
import numpy as np
# import energyusage
from codecarbon import OfflineEmissionsTracker


rng = np.random.default_rng(42)


def measure_time(function):
    """Measure CPU and wall times of a given function."""
    gc.collect()
    tracker = OfflineEmissionsTracker(
        tracking_mode = "process",
        measure_power_secs = 0.1,
        save_to_file = False,
        save_to_api = False,
        save_to_logger = False,
        country_iso_code = "FIN",
        log_level = "critical",
    )
    tracker.start()
    t1 = time.perf_counter(), time.process_time()

    # TODO: This stuff was for the PR version of the library but it doesn't seem to work. 
    # os.environ["PAPI_EVENTS"] = "PAPI_TOT_CYC"
    # os.environ["PAPI_OUTPUT_DIRECTORY"] = "."
    # papi_high.hl_region_begin("computation")
    # TODO: For some reason perf counters are not available even though `papi_avail` shows that some, including `PAPI_TOT_CYC`, are.
    # print(papi_high.num_counters())
    # papi_high.start_counters([
    #     papi_events.PAPI_TOT_CYC
    # ])
    # _, energy, _ = energyusage.evaluate(function, printToScreen=True, energyOutput=True)
    function()
    # TODO: For PR PAPI
    # papi_high.hl_read("computation")
    # papi_high.hl_region_end("computation")
    # TODO: For released PAPI
    # [cycles] = papi_high.stop_counters()
    t2 = time.perf_counter(), time.process_time()
    co2 = tracker.stop()
    counter_diff = t2[0] - t1[0]
    cpu_time = t2[1] - t1[1]
    return co2, counter_diff, cpu_time


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
                    co2, counter, cpu_time = measure_time(
                        lambda: library.groupby(sdf, groupby_column)
                    )
                case "sort":
                    co2, counter, cpu_time = measure_time(
                        lambda: library.sort_column(sdf, sort_column)
                    )
                case "drop_duplicates":
                    co2, counter, cpu_time = measure_time(lambda: library.drop_duplicates(sdf))
            res[ti, 3 * tj + 0] = co2
            res[ti, 3 * tj + 1] = counter
            res[ti, 3 * tj + 2] = cpu_time
    res_df = pd.DataFrame(res)
    res_df.columns = [prefix + "_" + library.method_name + "_" + postfix for postfix in tests for prefix in ["CO2", "cntr", "cpu"]]
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
