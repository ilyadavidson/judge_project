"""
Parse merged OpenAI batch results and attach to case rows.
Writes:
  - artifacts/api_answers.parquet  (per-case model outputs keyed by custom_id)
  - artifacts/cl_with_answers.parquet
  - artifacts/cases_all.parquet     (CL + CAP where applicable)
"""
from pathlib import Path
import argparse
import pandas as pd

from api_call import load_case_results  # parses JSONL -> tidy DF with custom_id + 9 keys

DATA_DIR       = Path("data")
ARTIFACTS_DIR  = Path("artifacts")
BATCH_DIR      = Path("batch_runs")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main(args):
    # 1) parse the big merged responses JSONL (api_call.py keeps it at batch_runs/api_responses.jsonl)
    results_jsonl = BATCH_DIR / "api_responses.jsonl"
    answers = load_case_results(str(results_jsonl))
    if answers.empty:
        print("[API] no parsed answers found.")
        return

    # answers.custom_id corresponds to CL 'unique_id' without CL_ prefix
    answers = answers.copy()
    answers["custom_id"] = answers["custom_id"].astype(str)

    # 2) attach to CL
    cl = pd.read_csv(ARTIFACTS_DIR / "cl_data_clean.csv")
    cl = cl.copy()
    cl["custom_id"] = cl["unique_id"].astype(str).str.replace("^CL_", "", regex=True)

    cl_ans = cl.merge(answers, on="custom_id", how="left")

    # 3) optional: stitch with CAP enriched
    cap = pd.read_parquet(ARTIFACTS_DIR / "cap_enriched.parquet")

    # choose a join key (docket_number often best for CAP↔CL if present)
    cases_all = cl_ans.merge(
        cap, on="docket_number", how="outer", suffixes=("_cl","_cap")
    )

    answers.to_parquet(ARTIFACTS_DIR / "api_answers.parquet", index=False)
    cl_ans.to_parquet(ARTIFACTS_DIR / "cl_with_answers.parquet", index=False)
    cases_all.to_parquet(ARTIFACTS_DIR / "cases_all.parquet", index=False)

    print(f"[API] answers: {len(answers):,} | cl_with_answers: {len(cl_ans):,} | cases_all: {len(cases_all):,}")

if __name__ == "__main__":
    _ = argparse.ArgumentParser().parse_args()
    main(_)