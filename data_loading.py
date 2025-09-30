import os, re, glob
import numpy as np
import pandas as pd

import pyarrow.dataset as ds
import pyarrow.compute as pc
import tiktoken

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
        pattern     = "data/parquet_files/CAP_data_*.parquet",
        appellate   = ["Third Circuit"], 
        district    = ["Delaware", "New Jersey", "Pennsylvania", "Virgin Islands"]):
    """
    Builds the dataframe off of the parquet files. Due to storage there's an option to only load in certain circuits.
    Default is set to the third circuit.
    Put either appellate or district as None to get the whole df. 
    """
    
    # Loads in the parquet files
    ############################################################################################################
    files = sorted(glob.glob(pattern))
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

    df = table.to_pandas(use_threads=True)

    # Creates an unique id and appellate identifier
    ############################################################################################################
    df["unique_id"] = df.index.astype(str)
    df["is_appellate"] = np.where(df["court_name"].str.contains("Appeals", case=False, na=False), 1, 0)
    return df

enc = tiktoken.get_encoding("o200k_base")

def truncate_opinion(text, max_tokens= 6000) -> str:
    text = "" if text is None else str(text)
    toks = enc.encode(text)
    if len(toks) > max_tokens:
        head = toks[:max_tokens]
        tail = toks[-max_tokens:]
        toks = head + tail
    return enc.decode(toks)

def text_builder(df, limit, mx_tk):
    """ 
    Function to call in dataset.
    
    :param df: original df.
    :param limit: how many cases to load.
    :param max_tokens_each: how many of the last tokens we want to keep.
    """
    if limit    == None:
        subset  = df[df['is_appellate']==1].copy()
    else:
        subset  = df[df['is_appellate']==1].head(limit).copy()
    
    results = []

    for _, row in subset.iterrows():
        cid         = row["unique_id"]
        raw_text    = row["opinion_text"]
        trimmed     = truncate_opinion(raw_text, max_tokens=mx_tk)
        results.append({"id": cid, "text": trimmed})

    return results