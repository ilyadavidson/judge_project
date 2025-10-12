"""
This file handles everything that has to do with getting the CAP dataset out of the data/parquet files and getting it prepared for the rest of the repo. 
"""

import os, re, glob
import numpy as np
import pandas as pd

import pyarrow.dataset as ds
import pyarrow.compute as pc

from pathlib import Path


def _arrow_filter_table(files: list[str], courts: list[str]):
    """
    Use PyArrow Dataset to scan many Parquet files while filtering over smaller subset.

    :param files: parquet files.
    :param courts: courts included in the final df.
    """
    dataset         = ds.dataset(files, format="parquet")

    courts_regex    = rf"(?i)({'|'.join(map(re.escape, courts))})"
    filt            = pc.match_substring_regex(ds.field("court_name"), courts_regex)

    table           = dataset.to_table(filter=filt, use_threads=True)
    return table

def build_cap_dataset(
        parquet_root: str | Path         = "data/parquet_files",
        appellate: list[str] | None      = ["Third Circuit"], 
        district: list[str]  | None      = ["Delaware", "New Jersey", "Pennsylvania", "Virgin Islands"]):
    """
    Builds the dataframe off of the parquet files. Due to storage there's an option to only load in certain circuits.
    Default is set to the third circuit.
    Put either appellate or district as None to get the whole df. 
    """
    
    # Loads in the parquet files
    ############################################################################################################
    parquet_root = Path(parquet_root)
    pattern = parquet_root / "CAP_data_*.parquet"

    files = sorted(glob.glob(str(pattern)))
    print(f"Working dir: {os.getcwd()}")
    print(f"Found {len(files)} parquet files for pattern: {pattern}")

    # Make a sub-set if needed
    ############################################################################################################
    if appellate is None or district is None:
        courts      = None
        dataset     = ds.dataset(files, format="parquet")
        table       = dataset.to_table(use_threads=True)
    else:
        courts      = appellate + district
        table       = _arrow_filter_table(files, courts=courts)

    if table.num_rows == 0:
        return pd.DataFrame()

    cols = ["name", "docket_number", "decision_date", "court_name", "opinion_author_clean", "opinion_author_id", "opinion_text"]
    df = table.to_pandas(use_threads=True)

    # Creates an unique id and appellate identifier
    ############################################################################################################
    df["unique_id"]         = df.index.astype(str)
    df["is_appellate"]      = np.where(df["court_name"].str.contains("Appeals", case=False, na=False), 1, 0)

    # Delete applicate duplicates based on opinion type and docket number and keeps the one with majority vote
    ############################################################################################################
    appellate = (
        df[df["is_appellate"] == 1]
        .sort_values("opinion_type", key=lambda s: s.str.lower().eq("majority"), ascending=False)
        .drop_duplicates(subset=["docket_number"], keep="first")
    )

    non_appellate = df[df["is_appellate"] != 1]
    df            = pd.concat([appellate, non_appellate], ignore_index=True)

    return df