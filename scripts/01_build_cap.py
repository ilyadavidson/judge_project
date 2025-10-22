"""
Build the CAP dataset (district + appellate), do basic filtering, and save.
"""
from    pathlib                 import Path
import  pandas                  as pd

from jp.cap.data_loading        import build_cap_dataset
from jp.cap.linking             import match_appellates
from jp.utils.constants         import circuits_to_district
from jp.utils.text              import ensure_dir

DATA_DIR        = Path("data")
CAP_DIR         = ensure_dir(DATA_DIR / "artifacts" / "cap")
CIRCUITS        = ensure_dir(CAP_DIR / "circuits")
MAPPING         = ensure_dir(CAP_DIR / "mapping")
out_path        = CIRCUITS / "cap_dataset.parquet"

def main(which):
    
    # 1. Building CAP dataset from parquet files
    ###############################################################################
    print("[CAP] building dataset…")

    pick = list((which or circuits_to_district.keys()))
    for cid in pick:
        path        = CIRCUITS / "all" / f"{cid}_cap.parquet"
        if path.exists():
            print(f"[CAP] Found {cid} dataset")
            continue
        df          = build_cap_dataset(parquet_root= DATA_DIR / "parquet_files", 
                                   appellate = circuits_to_district[cid]["appellate"], 
                                   district = circuits_to_district[cid]["district"]) 
        df.to_parquet(path, index=False)
        print(f"[CAP] Wrote {cid} ({len(df):,} rows)")

    out_files       = [CIRCUITS / "all" / f"{cid}_cap.parquet" for cid in pick if (CIRCUITS / "all" / f"{cid}_cap.parquet").exists()]
    out             = pd.concat((pd.read_parquet(f) for f in out_files), ignore_index=True)

    out_name        = "cap_dataset.parquet" if which is None else f"cap_{'_'.join(pick)}.parquet"
    out_path        = CIRCUITS / "used" / out_name
    out.to_parquet(out_path, index=False)
    print(f"[CAP] rows: {len(out):,}") 

    # 2. Appellate mapping to district cases and get judge. 
    # This is done per circuit to allow for circuit-specific matching logic.
    # ##############################################################################

    matched_parts = []
    for cid in pick:
        df_circuit   = pd.read_parquet(CIRCUITS / "all" / f"{cid}_cap.parquet")
        json_path    = MAPPING / f"appellate_matches_{cid}.json"
        matched_path = MAPPING / f"matched_{cid}.parquet"

        if json_path.exists() and matched_path.exists():
            print(f"[MAP] Found existing mapping for {cid}, skipping rematch.")
            matched_circuit = pd.read_parquet(matched_path)
        else:
            matched_circuit = match_appellates(df_circuit, path=str(json_path), re_match=True)
            matched_circuit.to_parquet(matched_path, index=False)
            print(f"[MAP] {cid}: wrote {json_path.name} and {matched_path.name} ({len(matched_circuit):,} rows)")

        matched_parts.append(matched_circuit)
    
    df_matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    print(f"[MAP] combined matched rows: {len(df_matched):,}")

    # 3.1. Ensure required columns are present
    required_columns     = {"district judge id", "district judge", "opinion_text", "unique_id", "name", "docket_number"}
    missing_columns      = required_columns - set(df_matched.columns)

    if missing_columns:
        raise ValueError(f"[CAP] Missing required columns: {', '.join(missing_columns)}")

    # 3. Return clean CAP files.
    ##############################################################################
    matched_name = "cap_matched.parquet" if which is None else f"cap_matched_{'_'.join(pick)}.parquet"
    out_parquet  = CAP_DIR / matched_name

    df_matched.to_parquet(out_parquet, index=False)
    print(f"[CAP] wrote {out_parquet} ({len(df_matched):,} rows)")

if __name__ == "__main__":
    main(["1st", "2nd", "3rd"]) # set to None to do all circuits