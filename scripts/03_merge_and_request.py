"""
Merge CAP & CL at docket level (for backfilling judge IDs), then
enqueue OpenAI Batch requests **only for cases not yet answered**.
"""
from pathlib import Path
import argparse
import pandas as pd

from api_call import run_incremental_batches

DATA_DIR       = Path("data")
ARTIFACTS_DIR  = Path("artifacts")
BATCH_DIR      = Path("batch_runs")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

def _prep_for_api(cl: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a frame with:
      - id: string without 'CL_' prefix
      - opinion_text
    De-dupe on id to avoid resending.
    """
    df = cl.copy()
    if "id" not in df.columns:
        df["id"] = df["unique_id"].astype(str).str.replace("^CL_", "", regex=True)
    df = df[["id", "opinion_text"]].dropna(subset=["opinion_text"])
    df = df.drop_duplicates(subset="id", keep="first")
    return df

def main(args):
    cap = pd.read_parquet(ARTIFACTS_DIR / "cap_clean.parquet")
    cl  = pd.read_csv(ARTIFACTS_DIR / "cl_data_clean.csv")

    # optional: backfill CAP judge ids from CL where dockets match and CAP is missing
    # (simple fill; your heavier logic can live in results.merge_cap_and_cl if you prefer)
    cap = cap.copy()
    cl_d = cl[["docket_number","district judge","district judge id"]].dropna(subset=["docket_number"])
    cap = cap.merge(cl_d, on="docket_number", how="left", suffixes=("","_cl"))
    for col in ["district judge","district judge id"]:
        if col in cap.columns and f"{col}_cl" in cap.columns:
            cap[col] = cap[col].fillna(cap[f"{col}_cl"])
            cap = cap.drop(columns=[f"{col}_cl"])

    cap.to_parquet(ARTIFACTS_DIR / "cap_enriched.parquet", index=False)
    print(f"[Merge] wrote cap_enriched.parquet ({len(cap):,} rows)")

    # Build and send **incremental** requests from CL
    df_for_api = _prep_for_api(cl)
    print(f"[Batch] ready to consider {len(df_for_api):,} CL opinions for API")

    run_incremental_batches(
        df_for_api,
        work_dir="batch_runs",
        full_input_name="overlap_input2.jsonl",
        missing_input_name="overlap_input2_missing.jsonl",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        max_bytes=200*1024*1024,
        poll=args.poll,                   # pass --poll if you want to wait
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--poll", action="store_true", help="Poll the batch to completion and download results now")
    main(p.parse_args())