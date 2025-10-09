"""
Build the CAP dataset (district + appellate), do basic filtering, and save.
"""
from pathlib import Path
import argparse
import pandas as pd

# your modules
from cap.data_loading import build_cap_dataset, keep_majority_for_appellate

# if yours live elsewhere: from data_loading import build_cap_dataset, keep_majority_for_appellate

DATA_DIR       = Path("data")
ARTIFACTS_DIR  = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main(args):
    print("[CAP] building dataset…")
    df = build_cap_dataset(parquet_root=DATA_DIR / "parquet_files")
    print(f"[CAP] rows: {len(df):,}")

    print("[CAP] keep majority opinion for appellate cases…")
    df = keep_majority_for_appellate(df)

    out_parquet = ARTIFACTS_DIR / "cap_clean.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[CAP] wrote {out_parquet} ({len(df):,} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main(_)