import pandas as pd
import numpy as np

def compute_overturns(judges: pd.DataFrame, full_data: pd.DataFrame) -> pd.DataFrame:
    df = full_data.copy()
    df["judge id"]      = pd.to_numeric(df["district judge id"], errors="coerce").astype("Int64")
    df["opinion"]       = df["opinion"].astype(str).str.lower()
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

    j = judges.copy()
    j["promotion_date"] = pd.to_datetime(j["promotion_date"], errors="coerce")
    j["cutoff"]         = j["promotion_date"] - pd.DateOffset(months=3)

    m = df.merge(j[["judge id","cutoff"]], on="judge id", how="inner")
    m = m[m["decision_date"].notna() & (m["cutoff"].isna() | (m["decision_date"] <= m["cutoff"]))]

    counts = m.groupby("judge id", dropna=True).agg(
        appealed_cases=("unique_id","count"),
        overturned_appealed_cases=("opinion", lambda x: (x != "affirmed").sum())
    )

    out = j.merge(counts, on="judge id", how="left")
    out[["appealed_cases","overturned_appealed_cases"]] = out[["appealed_cases","overturned_appealed_cases"]].fillna(0).astype(int)
    out["overturnrate"] = np.where(out["appealed_cases"] > 0,
                                   out["overturned_appealed_cases"] / out["appealed_cases"], np.nan)
    return out["overturnrate"]