"""
Aggregate per-judge features from case-level data (incl. API outputs).
Outputs:
  - artifacts/judges_features.parquet
"""
import pandas               as pd
from pathlib                import Path

from jp.features.engineer   import compute_overturns, prestige_calculator, us_support_calculator, politicality_calculator, citation_calculator
from jp.features.labels     import is_promoted

ARTIFACTS_DIR = Path("data/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():

    # 1. Making of feature dataset
    #####################################################################################
    c       = pd.read_parquet(ARTIFACTS_DIR / 'merged' / "cases_with_answers.parquet")
    j       = pd.read_csv("data/judge_info.csv")

    j['is promoted']                                   = is_promoted(j) 
    j['overturn rate']                                 = compute_overturns(j, c, cutoff=3) 
    j['prestige index']                                = prestige_calculator(j)
    j['US support rate'], j['US_support_N']            = us_support_calculator(j, c)
    j['politicality']                                  = politicality_calculator(j, c)
    j['citation_impact']                               = citation_calculator(j, c)

    keep_cols = ["judge id", "is promoted", "overturn rate", "prestige index", "aba rating", "gender", "ethnicity", "party of appointing president"]
    j         = j[[col for col in keep_cols if col in j.columns]]

    # 2. Output of feature dataset
    #####################################################################################
    out = ARTIFACTS_DIR / 'features' / "features.csv"

    j.to_csv(out, index=False)
    print(f"[Features] wrote {out} ({len(j):,} judges)")

if __name__ == "__main__":
    main()