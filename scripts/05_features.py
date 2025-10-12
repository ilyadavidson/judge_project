"""
Aggregate per-judge features from case-level data (incl. API outputs).
Outputs:
  - artifacts/judges_features.parquet
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from src.jp.features.engineer   import compute_overturns
from src.jp.features.labels     import promotion_info_judges

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main(args):

    # 1. Making of feature dataset
    #####################################################################################
    cases       = pd.read_parquet(ARTIFACTS_DIR / "cases_with_answers.parquet")
    judges      = pd.read_csv("data/judge_info.csv")

    judges['is_promoted']   = promotion_info_judges(judges) 
    judges['overturn_rate'] = compute_overturns(judges, cases)

    keep_cols = ["judge id", "is_promoted", "overturn_rate", "aba rating", "gender", "ethnicity"]
    judges = judges[[c for c in keep_cols if c in judges.columns]]

    # 2. Output of feature dataset
    #####################################################################################
    out = ARTIFACTS_DIR / "features.csv"

    judges.to_csv(out, index=False)
    print(f"[Features] wrote {out} ({len(judges):,} judges)")

if __name__ == "__main__":
    _ = argparse.ArgumentParser().parse_args()
    main(_)