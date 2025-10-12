"""
Build the CAP dataset (district + appellate), do basic filtering, and save.
"""
from    pathlib                 import Path
import  pandas                  as pd

from jp.cap.data_loading    import build_cap_dataset
from jp.cap.linking         import match_appellates

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
    
    df = pd.concat([df_3d, df_4d], ignore_index=True)

    print(f"[CAP] rows: {len(df):,}")

    # 2. Appellate mapping to district cases and get judge
    ########################################################
    df      = match_appellates(df, path="data/artifacts/cap/appellate_matches.json", re_match=True)

    print(f"Appellate matching done, kept {len(df)} cases.")

    # 3.1. Ensure required columns are present
    required_columns        = {"district judge id", "district judge", "opinion_text", "unique_id", "name", "docket_number"}
    missing_columns         = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"[CAP] Missing required columns: {', '.join(missing_columns)}")

    # 3. Return clean CAP files.
    ########################################################
    out_parquet = ARTIFACTS_DIR / "cap_dataset.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[CAP] wrote {out_parquet} ({len(df):,} rows)")

if __name__ == "__main__":
    main()