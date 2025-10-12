"""
Scrape / load CourtListener Third Circuit cases, extract district judges,
and save a clean CL dataset.
"""
import os
import pandas           as pd

from pathlib            import Path
from jp.cl.extract      import cl_loader, scrape_third_circuit

DATA_DIR       = Path("data")
ARTIFACTS_DIR  = DATA_DIR / Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Get judges info and load CL data or scrape CL.
    ########################################################
    judges          = pd.read_csv(DATA_DIR / "judge_info.csv")
    CSV_PATH        = "data/artifacts/cl/third_circuit_cases.csv"

    if os.path.exists(CSV_PATH):
        print(f"[CL] Loading existing {CSV_PATH}...")                           # load
        cl_data = pd.read_csv(CSV_PATH)
    else:
        print(f"[CL] {CSV_PATH} not found — scraping Third Circuit cases...")   # or scrape
        cl_data = scrape_third_circuit(limit=None, out_csv=CSV_PATH)

    cl          = cl_loader(cl_data, judges)                                    # extract judges info 

    # 2.1. Ensure required columns are present
    required_columns        = {"district judge id", "district judge", "opinion_text", "unique_id", "name", "docket_number"}
    missing_columns         = required_columns - set(cl.columns)

    if missing_columns:
        raise ValueError(f"[CAP] Missing required columns: {', '.join(missing_columns)}")

    # 2. Return clean CL files.
    ########################################################
    out_csv = ARTIFACTS_DIR / "cl" / "cl_data_clean.csv"
    cl.to_csv(out_csv, index=False)
    print(f"[CL] wrote {out_csv} ({len(cl):,} rows)")

if __name__ == "__main__":
    main()