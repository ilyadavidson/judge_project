"""
Scrape / load CourtListener Third Circuit cases, extract district judges,
and save a clean CL dataset.
"""
import os
import pandas           as pd

from pathlib            import Path
from jp.cl.scrape       import scrape_third_circuit
from jp.cl.extract      import cl_loader
from jp.utils.text      import ensure_dir
from jp.utils.constants import circuits

DATA_DIR        = Path("data")
ARTIFACTS_DIR   = ensure_dir(DATA_DIR / Path("artifacts"))
CL_DIR          = ensure_dir(ARTIFACTS_DIR / "cl")  
SCRAPED_DIR     = ensure_dir(CL_DIR / "scraped")  
CLEANED_DIR     = ensure_dir(CL_DIR / "cleaned")

def main(which, resume):
    # 1. Get judges info and load CL data or scrape CL.
    ########################################################
    # judges          = pd.read_csv(DATA_DIR / "judge_info.csv")
    
    if which is None:
        pick        = [c for c in circuits() if c not in {"dc","fed"}]
    else:
        pick        = [str(w).strip().lower() for w in which]

    parts = []
    for cid in pick:
        scraped_csv         = SCRAPED_DIR / f"{cid}_scraped.csv"
        if scraped_csv.exists() and not resume:
            print(f"[CL] Loading existing {scraped_csv}...")                           # load
            raw             = pd.read_csv(scraped_csv)
        else:
            print(f"[CL]s {scraped_csv} not found — scraping {cid} Circuit cases...")   # or scrape
            raw             = scrape_third_circuit(cid = cid)

        # cleaned             = cl_loader(raw, judges)
        per_circuit_clean   = CLEANED_DIR / f"{cid}_cl_data_clean.csv" 

        # cleaned.to_csv(per_circuit_clean, index=False)
        # print(f"[CL] wrote {per_circuit_clean} ({len(cleaned):,} rows)")
        # parts.append(cleaned)

    # 2.1. Ensure required columns are present
    cl          = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    required    = {"district judge id","district judge","opinion_text","unique_id","name","docket_number"}
    missing     = required - set(cl.columns)
    if missing:
        raise ValueError(f"[CL] Missing required columns: {', '.join(missing)}")

    # 2. Return clean CL files.
    ########################################################
    out_name = "cl_data_clean.csv" if which is None else f"cl_data_clean_{'_'.join(pick)}.csv"
    out_csv  = CL_DIR / out_name
    cl.to_csv(out_csv, index=False)
    print(f"[CL] wrote {out_csv} ({len(cl):,} rows)")

if __name__ == "__main__":
    main(["4th"], resume=True)  # set to None to do all circuits