"""
Aggregate per-judge features from case-level data (incl. API outputs).
Outputs:
  - artifacts/judges_features.parquet
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def _opinion_to_binary_overturn(opinion: str) -> float | None:
    """
    Map model 'opinion' (affirmed/reversed/vacated/remanded/...) to 1 for 'overturned-ish', 0 for affirmed.
    """
    if pd.isna(opinion): return None
    o = str(opinion).strip().lower()
    if o == "affirmed": return 0.0
    if o in {"reversed","vacated","remanded","reversed and remanded","modified"}: return 1.0
    return None

def main(args):
    cl = pd.read_parquet(ARTIFACTS_DIR / "cl_with_answers.parquet")

    # Use CL district judge id as the judge key
    if "district judge id" not in cl.columns:
        raise RuntimeError("cl_with_answers.parquet missing 'district judge id'")

    # Overturn indicator
    cl["overturn"] = cl["opinion"].map(_opinion_to_binary_overturn)

    # Basic per-judge aggregations
    grp = cl.groupby("district judge id", dropna=True)

    feats = grp.agg(
        n_cases              = ("overturn", "size"),
        n_overturned        = ("overturn", lambda s: np.nansum(s)),
        overturn_rate       = ("overturn", lambda s: np.nanmean(s)),
        first_case_date     = ("decision_date", "min"),
        last_case_date      = ("decision_date", "max"),
    ).reset_index()

    # Optionally merge static judge info (court, gender, ethnicity, etc.)
    try:
        judges = pd.read_csv("data/judge_info.csv")
        keep = ["judge id","first name","last name","court name","gender","ethnicity","is_promoted"]
        have = [c for c in keep if c in judges.columns]
        judges = judges[have].rename(columns={"judge id":"district judge id"})
        feats = feats.merge(judges, on="district judge id", how="left")
    except FileNotFoundError:
        pass

    out = ARTIFACTS_DIR / "judges_features.parquet"
    feats.to_parquet(out, index=False)
    print(f"[Features] wrote {out} ({len(feats):,} judges)")

if __name__ == "__main__":
    _ = argparse.ArgumentParser().parse_args()
    main(_)