import pandas as pd
from typing import Dict, List, Optional
import json

def load_case_results(path: str = "data/artifacts/api/outputs/cl_api_answers.jsonl") -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            try:
                content = rec["response"]["body"]["choices"][0]["message"]["content"]
                obj = json.loads(content)  # parse the 9-key JSON
            except Exception:
                continue
            obj["custom_id"] = rec.get("custom_id")  # keep your custom_id
            records.append(obj)
    return pd.DataFrame.from_records(records)

def get_id_from_names(out: pd.DataFrame, judges: pd.DataFrame) -> pd.Series:
    """
    Vectorized match: (lower_judge_first, lower_judge_last) in `out`
    -> 'judge id' from `judges`. Returns an Int64 Series aligned to `out.index`.
    """
    # Guard if name columns are missing
    if "lower_judge_first" not in out.columns or "lower_judge_last" not in out.columns:
        return pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"), index=out.index)

    # Normalize judges names
    J = judges.copy()
    J["__fn"] = J["first name"].astype(str).str.strip().str.lower()
    J["__ln"] = J["last name"].astype(str).str.strip().str.lower()
    # Drop exact dup name pairs to avoid exploding matches
    J = J.drop_duplicates(subset=["__fn", "__ln"])

    # Normalize API-provided names
    L = out.copy()
    L["__fn"] = L["lower_judge_first"].astype(str).str.strip().str.lower()
    L["__ln"] = L["lower_judge_last"].astype(str).str.strip().str.lower()

    # Left-merge to get judge id
    M = L.merge(J[["__fn", "__ln", "judge id"]], on=["__fn", "__ln"], how="left")

    # Return aligned Series (nullable Int64)
    return pd.to_numeric(M["judge id"], errors="coerce").astype("Int64")

def attach_api_to_cl_clean(
    cl_clean:           pd.DataFrame,
    api_path:           str = "data/artifacts/api/outputs/api_answers.jsonl",
    judges:             str = "data/judge_info.csv",
    *,
    id_col:             str = "unique_id",
    case_name_col:      str = "name",
    opinion_text_col:   str = "opinion_text",
    docket_col:         str = "docket_number",
    decision_date_col:  str = "decision_date",
) -> pd.DataFrame:
    judges= pd.read_csv(judges)
    out = cl_clean.copy()
    out['district judge id'] = pd.NA
    out['district judge'] = pd.NA

    # Mirror unique_id -> custom_id so your merge block stays identical
    out["custom_id"] = (
    out["unique_id"]
    .astype("string")
)

    # === Use your exact merge logic ===
    api_answers                 = load_case_results(api_path)
    api_answers                 = api_answers.copy()
    api_keys: List[str]         = [c for c in api_answers.columns if c != "custom_id"]

    out                         = out.merge(api_answers, left_on="custom_id", right_on="custom_id", how="left")

    # get district judge ids and names from judges dataframe
    out['district judge'] = out['lower_judge_last'].apply(lambda x: x.lower().strip() if pd.notna(x) else "")
    out["district judge id"] = get_id_from_names(out, judges)
    
    mask = out["district judge id"].notna()
    out.loc[mask, "district judge"] = (
        out.loc[mask, "lower_judge_last"].astype(str).str.strip().str.lower()
    )


    if "district judge id" in out.columns:
        out["district judge id"] = pd.to_numeric(out["district judge id"], errors="coerce").astype("Int64")

    base_cols = [id_col, case_name_col, opinion_text_col, docket_col, "district judge", "district judge id", decision_date_col]

    exclude_cols = {"lower_judge_first", "lower_judge_last"}
    api_cols     = [c for c in api_keys if c in out.columns and c not in exclude_cols]

    keep_cols = [c for c in base_cols if c in out.columns] + api_cols
    return out[keep_cols].reset_index(drop=True)

def cap_data_cleaner(
    cap_df:             pd.DataFrame,
    mapping_path:       str = "data/artifacts/cap/appellate_matches2.json",   # appellate_custom_id -> district_unique_id
    api_path:           str = "batch_runs/api_responses.jsonl",
    *,
    id_col:             str = "unique_id",
    judge_name_col:     str = "opinion_author_clean",
    judge_id_col:       str = "opinion_author_id",
    case_name_col:      str = "name",
    docket_col:         str = "docket_number",
    opinion_text_col:   str = "opinion_text",
    decision_date_col:  str = "decision_date",
) -> pd.DataFrame:
    """
    Cleans CAP data and merges in district judge information and API answers.
    """

    app_to_dct             = _load_mapping(mapping_path)
    map_df                 = pd.DataFrame(list(app_to_dct.items()), columns=["custom_id", "district_uid"]).astype(str)

    # Keep only the original appellate cases present in mapping (keep their appellate metadata)
    out                    = cap_df.copy()
    out[id_col]            = out[id_col].astype(str)
    out                    = out[out[id_col].isin(map_df["custom_id"])].copy()

    # Attach appropriate district judge
    district_lookup = (
        cap_df[[id_col, judge_name_col, judge_id_col]]
        .drop_duplicates(subset=[id_col])
        .rename(columns={id_col:            "district_uid",
                         judge_name_col:    "district judge",
                         judge_id_col:      "district judge id"})
    )
    out = (
        out.merge(map_df, left_on=id_col, right_on="custom_id", how="left")
           .merge(district_lookup, on="district_uid", how="left")
    )

    # Get API answers
    api_answers                 = load_case_results(api_path) 
    api_answers                 = api_answers.copy()
    api_keys: List[str] = [c for c in api_answers.columns if c != "custom_id"]

    out                         = out.merge(api_answers, left_on="custom_id", right_on="custom_id", how="left")
    out[id_col]                 = "CAP_" + out["custom_id"].astype(str)

    out["district judge id"]    = pd.to_numeric(out["district judge id"], errors="coerce").astype("Int64") # 5.0 -> 5

    base_cols                   = [id_col, case_name_col, opinion_text_col, docket_col, "district judge", "district judge id", decision_date_col]

    exclude_cols                = {"lower_judge_first", "lower_judge_last"}
    api_cols                    = [c for c in api_keys if c in out.columns and c not in exclude_cols]

    keep_cols                   = [c for c in base_cols if c in out.columns] + api_cols

    return out[keep_cols].reset_index(drop=True)