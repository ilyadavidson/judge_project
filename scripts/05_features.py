"""
Aggregate per-judge features from case-level data (incl. API outputs).
Outputs:
  - artifacts/judges_features.parquet
"""
import pandas               as pd
from pathlib                import Path

from jp.features.engineer   import compute_overturns
from jp.features.labels     import promotion_info_judges

ARTIFACTS_DIR = Path("data/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():

    # 1. Making of feature dataset
    #####################################################################################
    cases       = pd.read_parquet(ARTIFACTS_DIR / 'merged' / "cases_with_answers.parquet")
    judges      = pd.read_csv("data/judge_info.csv")

    judges['is_promoted']   = promotion_info_judges(judges) 
    judges['overturn_rate'] = compute_overturns(judges, cases, cutoff=3) 

    keep_cols = ["judge id", "is_promoted", "overturn_rate", "aba rating", "gender", "ethnicity"]
    judges = judges[[c for c in keep_cols if c in judges.columns]]

    # 2. Output of feature dataset
    #####################################################################################
    out = ARTIFACTS_DIR / 'features' / "features.csv"

    judges.to_csv(out, index=False)
    print(f"[Features] wrote {out} ({len(judges):,} judges)")

if __name__ == "__main__":
    main()