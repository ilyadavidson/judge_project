"""
Scrape / load CourtListener Third Circuit cases, extract district judges,
and save a clean CL dataset.
"""
from pathlib import Path
import argparse
import pandas as pd

from scr.jp.cl.extract import cl_loader

DATA_DIR       = Path("data")
ARTIFACTS_DIR  = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main(args):
    judges = pd.read_csv(DATA_DIR / "judge_info.csv")
    print(f"[CL] judges loaded: {len(judges):,}")

    print("[CL] loading / scraping third circuit cases…")
    cl = cl_loader(judges)

    # ensure minimal required columns exist
    need = {'name','docket_number','decision_date','opinion_text',
            'district judge','district judge id','is_appellate','unique_id','overlap_by_substring'}
    missing = need - set(cl.columns)
    if missing:
        raise RuntimeError(f"[CL] missing columns: {missing}")

    # keep only rows with usable text
    cl = cl[cl["opinion_text"].notna()].copy()

    out_csv = ARTIFACTS_DIR / "cl_data_clean.csv"
    cl.to_csv(out_csv, index=False)
    print(f"[CL] wrote {out_csv} ({len(cl):,} rows)")

if __name__ == "__main__":
    main()