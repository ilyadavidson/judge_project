"""
Build the CAP dataset (district + appellate), do basic filtering, and save.
"""
from    pathlib                 import Path
import  pandas                  as pd

from jp.cap.data_loading        import build_cap_dataset
from jp.cap.linking             import match_appellates

DATA_DIR       = Path("data")
ARTIFACTS_DIR  = DATA_DIR / "artifacts" / "cap"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    
    # 1. Building CAP dataset from parquet files
    ########################################################
    print("[CAP] building dataset…")

    df_3d      = build_cap_dataset(parquet_root=DATA_DIR / "parquet_files",  # Third circuit
                                   appellate = ["Third Circuit"], 
                                   district = ["Delaware", "New Jersey", "Pennsylvania", "Virgin Islands"])
    df_4d      = build_cap_dataset(parquet_root=DATA_DIR / "parquet_files",  # Fourth circuit
                                   appellate = ["Fourth Circuit"], 
                                   district = ["Maryland", "Carolina", "Virginia"])

    print(f"[CAP] rows: {len(df_3d) + len(df_4d):,}") # in two go's for memory reasons

    # 2. Appellate mapping to district cases and get judge. 
    # This is done per circuit to allow for circuit-specific matching logic.
    ########################################################

    results = {}
    for df_name, df in [("3d", df_3d), ("4d", df_4d)]:
        results[df_name]      = match_appellates(df, path="data/artifacts/cap/appellate_matches_{df_name}.json", re_match=True)

    df_matched = pd.concat(results.values(), ignore_index=True)
    print(f"Appellate matching done, kept {len(df_matched)} cases.")

    # 3.1. Ensure required columns are present
    required_columns        = {"district judge id", "district judge", "opinion_text", "unique_id", "name", "docket_number"}
    missing_columns         = required_columns - set(df_matched.columns)

    if missing_columns:
        raise ValueError(f"[CAP] Missing required columns: {', '.join(missing_columns)}")

    # 3. Return clean CAP files.
    ########################################################
    out_parquet = ARTIFACTS_DIR / "cap_dataset.parquet"
    df_matched.to_parquet(out_parquet, index=False)
    print(f"[CAP] wrote {out_parquet} ({len(df_matched):,} rows)")

if __name__ == "__main__":
    main()