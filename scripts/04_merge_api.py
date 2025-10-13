"""
At this point you have a file of cases (part CL, part CAP) and a big JSONL of OpenAI responses.
This file merges the answers to the file of cases.

Needs:
  - artifacts/api_responses.jsonl  (per-case model outputs keyed by custom_id)
  - artifacts/cl_with_answers.parquet
  - artifacts/cases_all.parquet     (CL + CAP where applicable)
"""

from    pathlib     import Path
import  argparse
import  pandas      as pd

from src.jp.api.results import attach_api_to_cl_clean, cap_data_cleaner, load_case_results  # match judge names to judge ids

DATA_DIR       = Path("data")
ARTIFACTS_DIR  = Path("data/artifacts")
API_OUTPUT_DIR = Path("data/artifacts/api/outputs")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def merge_cap_and_cl(cap: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    cap = cap.copy()
    cl  = cl.copy()

    # Normalize dockets for reliable matching
    cap["_dkey"] = cap["docket_number"].astype(str).str.strip()
    cl["_dkey"]  = cl["docket_number"].astype(str).str.strip()

    # Build fast lookup (first non-null per docket) for IDs and names from CL
    cl_id_map   = (cl[["_dkey", "district judge id"]]
                   .dropna(subset=["district judge id"])
                   .drop_duplicates("_dkey")
                   .set_index("_dkey")["district judge id"])
    cl_name_map = (cl[["_dkey", "district judge"]]
                   .dropna(subset=["district judge"])
                   .drop_duplicates("_dkey")
                   .set_index("_dkey")["district judge"])

    # Fill CAP's missing judge id/name from CL where dockets match
    miss_id = cap["district judge id"].isna()
    cap.loc[miss_id, "district judge id"] = cap.loc[miss_id, "_dkey"].map(cl_id_map)

    if "district judge" in cap.columns:
        miss_name = cap["district judge"].isna()
        cap.loc[miss_name, "district judge"] = cap.loc[miss_name, "_dkey"].map(cl_name_map)

    # Append CL rows that don't overlap by docket
    non_overlap_cl = cl[~cl["_dkey"].isin(cap["_dkey"])]
    non_overlap_cl = non_overlap_cl.reindex(columns=cap.columns, fill_value=np.nan)

    out = pd.concat([cap, non_overlap_cl], ignore_index=True)
    return out.drop(columns=["_dkey"], errors="ignore")

def main(args):
    """
    1. Load the big JSONL of OpenAI responses
    2.  If custom_id starts with CL: attach to CL data with district judge info and remove rows where info not found.
        If custom_id starts with CAP: if there is info on district judge, attach it, otherwise keep the row as is (with manual district judge lookup).
    3. Remove rows with no answers (e.g. if the model failed to respond) or no district judge. 
    """

    # 1. Load the API responses and the case data (CL and CAP)
    #######################################################################################
    print("[API] merging API results with case data...")

    results_jsonl           = API_OUTPUT_DIR / "api_responses.jsonl"
    answers                 = load_case_results(str(results_jsonl))
    answers["custom_id"]    = answers["custom_id"].astype(str)
        
    has_cl                  = answers["custom_id"].str.startswith("CL_").any()
    has_cap                 = answers["custom_id"].str.startswith("CAP_").any()

    if answers.empty:
        print("[API] no parsed answers found.")
        return
    
    cl                      = ARTIFACTS_DIR / "cl" / "cl_data_clean.csv"
    cap                     = ARTIFACTS_DIR / "cap" / "cap_dataset.parquet"

    # 2. If CL, attach to CL data with district judge info and remove rows where info not found.
    # If CAP, attach where info is found, but otherwise use our manual district judge lookup.
    #########################################################################################

    if has_cl:
        print("[API] Detected CourtListener (CL) results → attaching to CL dataset…")
        cl_clean = attach_api_to_cl_clean(cl)
    if has_cap:
        print("[API] Detected CAP results → cleaning CAP dataset…")
        cap_clean = cap_data_cleaner(cap)
    else:
        print("[API] No rows after cleaning/attachment. Exiting.")
        return
    
    full = merge_cap_and_cl(cap_clean, cl_clean)

    # 3. Return the full dataset with answers and district judge info
    ##########################################################################################
    out_csv = ARTIFACTS_DIR / "merged" / "cases_with_answers.csv"
    full.to_csv(out_csv, index=False)
    print(f"[CL] wrote {out_csv} ({len(cl):,} rows)")

if __name__ == "__main__":
    _ = argparse.ArgumentParser().parse_args()
    main(_)