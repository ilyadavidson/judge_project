"""
Merge CAP & CL at docket level (for backfilling judge IDs), then
enqueue OpenAI Batch requests **only for cases not yet answered**.
"""
import pandas       as pd
from pathlib        import Path

from jp.api.submit  import build_input, split_by_size, enqueue_chunks, download_new_outputs

ARTIFACTS_DIR  = Path("data/artifacts")
BATCH_DIR      = Path("batch_runs")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Merge the made datasets in the final dataset
    #######################################################################
    cl      = pd.read_parquet(ARTIFACTS_DIR  / "cl" / "cl_clean.parquet")
    cap     = pd.read_parquet(ARTIFACTS_DIR  / "cap" / "cap_clean.parquet")
    merged  = pd.concat([cap, cl], ignore_index=True)

    # 2. To avoid duplicate requests, check which cases have already been answered
    ########################################################################
    answered_ids = set()
    for out_path in ARTIFACTS_DIR.glob("all_api_answers.jsonl"):
        df = pd.read_json(out_path, lines=True)
        answered_ids.update(df["unique_id"].astype(str).tolist())

    print(f"Found {len(answered_ids)} answered cases in {ARTIFACTS_DIR}")

    merged["unique_id"]     = merged["unique_id"].astype(str)
    to_request              = merged[~merged["unique_id"].isin(answered_ids)]

    print(f"{len(to_request)} cases to request from API")

    # 3. Put the cases to request through the API pipeline
    #############################################################################
    input               = build_input(df, 'data/artifacts/api/requests/api_requests.jsonl') 
    parts               = split_by_size(input, output_path='batch_runs/input_chunks', prefix="input_chunk", max_bytes=200 * 1024 * 1024)
    batch_ids           = enqueue_chunks(parts, endpoint="/v1/chat/completions", completion_window="24h")
    _                   = download_new_outputs(batch_ids, output_path = 'data/api/outputs', poll=False)
# ============================================================================

if __name__ == "__main__":
    main()