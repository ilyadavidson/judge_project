"""
Merge CAP & CL at docket level (for backfilling judge IDs), then
enqueue OpenAI Batch requests **only for cases not yet answered**.
"""
import pandas       as pd
from pathlib        import Path

from jp.api.submit  import request_api, check_batch_status, download_batch_output_file, parse_batch_output

ARTIFACTS_DIR  = Path("data/artifacts")
BATCH_DIR      = Path("batch_runs")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Merge the made datasets in the final dataset
    #######################################################################
    cl      = pd.read_parquet(ARTIFACTS_DIR  / "cl" / "cl_data_clean.parquet")
    cap     = pd.read_parquet(ARTIFACTS_DIR  / "cap" / "cap_dataset.parquet")
    merged  = pd.concat([cap, cl], ignore_index=True)

    # 2. To avoid duplicate requests, check which cases have already been answered
    ########################################################################
    answered_ids = set()
    for out_path in ARTIFACTS_DIR.glob("all_api_answers.jsonl"):
        df = pd.read_json(out_path, lines=True)
        answered_ids.update(df["unique_id"].astype(str).tolist())
    print(f"Found {len(answered_ids)} answered cases in {ARTIFACTS_DIR}")

    merged["unique_id"] = merged["unique_id"].astype(str)
    to_request          = merged[~merged["unique_id"].isin(answered_ids)]
    print(f"{len(to_request)} cases to request from API")

    # 3. Put the cases to request through the API pipeline
    #########################################################################
    request_api(to_request)
    check_batch_status()
    download_batch_output_file()
    parse_batch_output()

if __name__ == "__main__":
    main()