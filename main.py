"""
Mapping appellate court cases to their respective lower court cases.
"""

import numpy    as np
import pandas   as pd
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing           import normalize
from typing                         import Callable

from helper_functions                import split_on_v, _find_docket_in_text, _norm_docket, _candidate_judge_names, _text_contains_any, normalize_case_name, norm_id
from helper_functions                import split_normalize_dockets, extract_all_dockets
from data_loading                    import build_cap_dataset
from api_call                        import _extract_text


# Mapping appellate judges to district judges
###################################################################################
def build_district_tfidf_index(     df:         pd.DataFrame = None, 
                                    nrm:        Callable = normalize_case_name, 
                                    analyzer:   str = "word", 
                                    min_df:     int = 1, 
                                    max_df:     float = 0.8, 
                                    side:       bool = False):
    """
    Build a TF-IDF index over all district cases.

    :param df:          DataFrame with CAP data, must include 'is_appellate', 'name', 'decision_date'
    :param nrm:         Function to normalize case names
    :param analyzer:    'word' or 'char' for TfidfVectorizer
    :param min_df:      Ignores terms that have a document frequency strictly lower than the given threshold. (If = 1 then used all terms).
    :param max_df:      Decides the maximum proportion of documents a term can appear in to be included (if <1, deletes common terms)
    :param side:        Builds either a single whole-caption TF-IDF matrix (False) or side-aware matrices (True) if cases will be split on vs./v. 
    """
    
    dcts        = df[df["is_appellate"] == 0].copy()    # district cases
    raw         = dcts["name"].astype(str)              # raw case names

    # Not side-aware branch
    #####################################################################################################################
    if not side:
        names   = raw.map(nrm) if nrm else raw
        vec     = TfidfVectorizer(
            analyzer    = analyzer,
            min_df      = min_df, 
            max_df      = max_df
        )
        X_dct   = vec.fit_transform(names)
        X_dct   = normalize(X_dct, norm="l2", copy=False)

        return {
            "vectorizer": vec,
            "X_dct": X_dct,
            "dct_index": dcts.index.to_numpy(),
            "dct_dates": pd.to_datetime(dcts["decision_date"], errors="coerce").to_numpy(),
            "dct_names": names.reset_index(drop=True),
        }

    # Side-aware branch
    ####################################################################################################################
    LR          = raw.apply(split_on_v)                                     # splits the case names into a plaintiff and defendent side
    has_both    = LR.map(lambda t: t[0] is not None and t[1] is not None)   # only keep those with both sides

    dcts        = dcts.loc[has_both].copy()
    left_raw    = LR.loc[has_both].map(lambda t: t[0])
    right_raw   = LR.loc[has_both].map(lambda t: t[1])

    left_norm   = left_raw.map(nrm) if nrm else left_raw
    right_norm  = right_raw.map(nrm) if nrm else right_raw

    vec = TfidfVectorizer(
        analyzer    = analyzer, 
        min_df      = min_df, 
        max_df      = max_df
    )
    vec.fit(pd.concat([left_norm, right_norm], ignore_index=True))

    X_left  = normalize(vec.transform(left_norm),  norm="l2", copy=False)
    X_right = normalize(vec.transform(right_norm), norm="l2", copy=False)

    return {
        "vectorizer": vec,
        "X_left": X_left,
        "X_right": X_right,
        "dct_index": dcts.index.to_numpy(),
        "dct_dates": pd.to_datetime(dcts["decision_date"], errors="coerce").to_numpy(),
        "left_norm": left_norm.reset_index(drop=True),
        "right_norm": right_norm.reset_index(drop=True),
        "left_raw": left_raw.reset_index(drop=True),
        "right_raw": right_raw.reset_index(drop=True),
    }

def appellate_mapping(df, side_index, nrm, score_cutoff, side_threshold, whole_index):
    """
    Side-aware TF-IDF matching for appellate cases. 

    :param df:              DataFrame with all cases
    :param side_index:      Output of build_side_aware_district_index()
    :param normalize_fn:    Function to normalize case names
    :param score_cutoff:    Minimum similarity score to report a match
    :param side_threshold:  Minimum side similarity (0-1) to consider both sides matching
    :param fallback_index:  Optional district index (output of build_district_tfidf_index) to use when no 'v' in appellate name.
    """

    # Obtain the side-aware district index built in function above
    ############################################################################################################
    vec       = side_index["vectorizer"]
    X_left    = side_index["X_left"]
    X_right   = side_index["X_right"]
    dct_idx   = side_index["dct_index"]
    dct_dates = side_index["dct_dates"]

    # Obtain the appellate cases to match
    ############################################################################################################
    apps            = df[df["is_appellate"] == 1].copy()
    app_raw_names   = apps["name"].astype(str)
    app_dates       = pd.to_datetime(apps["decision_date"], errors="coerce").to_numpy()

    # Find best match for each appellate case by first checking sides
    ############################################################################################################
    out = []
    for ai, a_raw, a_dt in zip(apps.index, app_raw_names, app_dates):
        L_raw, R_raw = split_on_v(a_raw)

        # If no 'v' in appellate: fallback to whole-caption TF-IDF
        ############################################################################################################
        if L_raw is None or R_raw is None:
            if not whole_index:
                continue  # no fallback provided
            # normalize whole caption & query
            q       = nrm(a_raw)
            xq      = normalize(whole_index["vectorizer"].transform([q]), norm="l2", copy=False)
            sims    = whole_index["X_dct"].dot(xq.T).toarray().ravel()

            # 7-year window (we make the assumption that appellate cases are within 7 years of district case).
            if pd.isna(a_dt):
                valid   = np.ones_like(sims, dtype=bool)
            else:
                a_ts    = pd.Timestamp(a_dt)
                lower   = (a_ts - pd.DateOffset(years=7)).to_datetime64()
                upper   = a_ts.to_datetime64()
                valid   = (whole_index["dct_dates"] >= lower) & (whole_index["dct_dates"] <= upper)
            if not valid.any():
                continue
            
            sims[~valid] = 0.0 # Set all the dates that don't match score to 0

            r       = int(np.argmax(sims))
            best    = float(sims[r])
            if best <= score_cutoff:
                continue
            out.append({
                "appellate_index":      ai,
                "district_index":       int(whole_index["dct_index"][r]),
                "score":                best + 0.15, # Since there's no v we add a small bonus to the score as it's more difficult to be similar
                "appellate_name":       q,
                "district_name":        whole_index["dct_names"].iloc[r],
                "appellate_name_raw":   a_raw,
                "district_name_raw":    df.at[int(whole_index["dct_index"][r]), "name"],
            })
            continue

        # If there's a v.: Normalize each side and apply TF-IDF. If either side scores below side_threshold, score=0.
        ############################################################################################################
        L_norm  = nrm(L_raw)
        R_norm  = nrm(R_raw)

        xL      = normalize(vec.transform([L_norm]), norm="l2", copy=False)
        xR      = normalize(vec.transform([R_norm]), norm="l2", copy=False)

        sims_L  = X_left.dot(xL.T).toarray().ravel()
        sims_R  = X_right.dot(xR.T).toarray().ravel()

        # 7-year window on district dates
        if pd.isna(a_dt):
            valid = np.ones_like(sims_L, dtype=bool)
        else:
            a_ts    = pd.Timestamp(a_dt)
            lower   = (a_ts - pd.DateOffset(years=7)).to_datetime64()
            upper   = a_ts.to_datetime64()
            valid   = (dct_dates >= lower) & (dct_dates <= upper)
        if not valid.any():
            continue

        sims_L[~valid] = 0.0
        sims_R[~valid] = 0.0

        # Combine: gate by side_threshold, then average
        min_s       = np.minimum(sims_L, sims_R)
        combined    = np.where(min_s < side_threshold, 0.0, 0.5 * (sims_L + sims_R)) #if either side is lower than the side_threshold, we don't use this case (e.g. Jones vs. US and James vs. US, the left side needs to match too).

        r           = int(np.argmax(combined))
        best        = float(combined[r])
        if best <= score_cutoff:
            continue

        out.append({
            "appellate_index":  ai,
            "district_index":   int(dct_idx[r]),
            "score":            best,
            "appellate_name":   f"{L_norm} | {R_norm}",
            "district_name":    f"{side_index['left_norm'].iloc[r]} | {side_index['right_norm'].iloc[r]}",
            "appellate_name_raw": a_raw,
            "district_name_raw": df.at[int(dct_idx[r]), "name"],
        })

    return pd.DataFrame(out, columns=["appellate_index","district_index","score","appellate_name_raw","district_name_raw"])

def confirm_midrange_matches(matches, df_all, score_low=0.50, score_high=0.80):
    """
    Calculates additional confirmation for midrange matches using docket numbers and judge names.

    :param best_matches: DataFrame output from appellate_mapping()
    :param df_all: DataFrame with all cases, must include 'opinion_text', 'docket_number', 'opinion_author_raw'
    :param score_low: lower bound of midrange scores to check
    :param score_high: upper bound of midrange scores to check
    """

    out = matches.copy()
    out["text_confirmed"] = False
    out["confirm_reason"] = pd.NA

    # focus on midrange scores
    mid = out[(out["score"] >= score_low) & (out["score"] <= score_high)]

    for i, row in mid.iterrows():
        ai = row["appellate_index"]
        di = row["district_index"]

        # pull texts and fields
        app_text = df_all.at[ai, "opinion_text"] if ai in df_all.index else ""
        d_docket = df_all.at[di, "docket_number"] if di in df_all.index else ""
        d_judge  = df_all.at[di, "opinion_author_raw"]  if di in df_all.index else ""

        # 1) Docket check
        app_snip_dockets = extract_all_dockets(app_text)  # from appellate opinion text
        if app_snip_dockets:
            app_set = set(app_snip_dockets)
            dct_set = set(split_normalize_dockets(str(d_docket)))
            if app_set & dct_set:  # intersection non-empty
                out.at[i, "text_confirmed"] = True
                out.at[i, "confirm_reason"] = "docket"
                continue # done with this row

        # 2) Judge-name fallback
        judge_needles = _candidate_judge_names(d_judge)
        if judge_needles and _text_contains_any(app_text, judge_needles):
            out.at[i, "text_confirmed"] = True
            out.at[i, "confirm_reason"] = "judge"
    
    return out

def run_appellate_linking(df, 
                          whole_index, 
                          side_index , 
                          nrm               = normalize_case_name, 
                          score_cutoff      = 0.0, 
                          side_threshold    = 0.7, 
                          score_low         = 0.5, 
                          score_high        = 0.8, 
                          out_json_path     = "appellate_matches.json"):
    """
    Orchestrates the full pipeline of appellate mapping.
    Returns (best_matches_df, confirmed_df).

    :param df: DataFrame with all cases
    :param nrm: function to normalize case names
    :param score_cutoff: minimum similarity score to report a match
    :param side_threshold: minimum side similarity (0-1) to consider both sides matching
    :param score_low: lower bound of midrange scores to check
    :param score_high: upper bound of midrange scores to check
    :param out_json_path: path to write JSON output with both tables
    """
    df = df.copy()

    # Core matching
    ############################################################################################################
    best_matches = appellate_mapping(
        df              = df,

        side_index      = side_index,
        whole_index     = whole_index,
        nrm             = nrm,
        
        score_cutoff    = score_cutoff,
        side_threshold  = side_threshold,
  
    )

    # partition by score
    high    = best_matches[best_matches['score']>= score_high].copy()
    mid     = best_matches[(best_matches["score"] >= score_low) & (best_matches["score"] < score_high)].copy()

    # Midrange confirmations
    ############################################################################################################
    confirmed = confirm_midrange_matches(
        matches         = mid,
        df_all          = df,
        score_low       = score_low,
        score_high      = score_high,
    )

    confirmed_only = confirmed[confirmed["text_confirmed"] == True]

    # Write JSON bundle with both tables
    ############################################################################################################
    combined_dict = {}

    for rec in high.to_dict(orient="records"):
        combined_dict[rec["appellate_index"]] = rec["district_index"]

    for rec in confirmed_only.to_dict(orient="records"):
        combined_dict[rec["appellate_index"]] = rec["district_index"]

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_dict, f, ensure_ascii=False, indent=2)

    return best_matches, confirmed_only

# Mapping judge promotions
################################################################################

def judges_promoted_from_district(judge_info):
    """
    Returns a df of all judges who got promoted from district to appellate courts. Their promotion date is the earliest nomination date that they got to the appellate court. 
    """
    aj = judge_info[judge_info['court type'].str.contains('Appeals|Circuit', case=False, na=False)]
    dj = judge_info[judge_info['court type'].str.contains('District')]
    pj = aj[aj['judge id'].isin(dj['judge id'])].copy()
    pj['nomination date'] = pd.to_datetime(pj['nomination date'], errors='coerce')
    pj = pj.sort_values(['judge id','nomination date']).drop_duplicates('judge id', keep='first')
    return pj

def compute_district_overturns(
    df:             pd.DataFrame,
    judge_info:     pd.DataFrame,
    api_path:       str = "batch_runs/api_responses.jsonl",
    mapping_path:   str = "results/appellate_matches.json",
) -> pd.DataFrame:
    """
    Computes district-level overturn rates for judges promoted from district to appellate courts.
    Returns a subset of `judge_info` with added columns for the overturn rates of promoted appellate judges.
    """

    # Initialization
    # List per judge what cases they've done, so we can iterrate over the cases and see which ones got overturned
    ####################################################################################################################
    pj                     = judges_promoted_from_district(judge_info)
    dsc                    = df[df['is_appellate']==0].copy()
    dsc["_author_id_norm"] = dsc["opinion_author_id"].map(norm_id)

    by_judge_cases = (
        dsc
        .dropna(subset=["_author_id_norm"])
        .groupby("_author_id_norm")
        .apply(lambda g: list(g.index))
        .to_dict()
    )

    pj["_judge_id_norm"]        = pj["judge id"].map(norm_id)
    pj["district_cases_list"]   = pj["_judge_id_norm"].map(by_judge_cases).apply(lambda x: x if isinstance(x, list) else [])
    pj["district cases"]        = pj["district_cases_list"].str.len().astype(int)

    # Load the appellate to district mapping and check which appellate cases were overturned
    ####################################################################################################################
    with open(mapping_path, "r", encoding="utf-8") as f:
        app_to_dct = json.load(f)

    # API info
    overturned_district_indices = set()
    with open(api_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            txt = _extract_text(rec.get("response", {}))
            if not txt:
                continue
            try:
                obj = json.loads(txt)  # 9-key JSON from model
            except json.JSONDecodeError:
                continue
            opinion = obj.get("opinion")
            if opinion is None:
                continue
            if opinion != "affirmed":
                app_id = str(rec.get("custom_id"))
                if app_id in app_to_dct:
                    try:
                        overturned_district_indices.add(int(app_to_dct[app_id]))
                    except Exception:
                        # ignore non-integer district indices
                        pass

    # Get the overturn counts per judge
    ####################################################################################################################
    def _count_overturned(case_list):
        if not case_list:
            return 0
        return sum(1 for cid in case_list if cid in overturned_district_indices)

    pj["district_cases_overturned"]         = pj["district_cases_list"].apply(_count_overturned).astype(int)
    pj["district_overturn_rate"]            = pj.apply(
        lambda r: (r["district_cases_overturned"] / r["district cases"]) if r["district cases"] else pd.NA,
        axis=1
    )
    pj.drop(columns=["_judge_id_norm"], inplace=True)

    return pj


if __name__ == "__main__":
    df                  = build_cap_dataset()
    judge_info          = pd.read_csv("data/judge_info.csv")
    # whole_index         = build_district_tfidf_index(df, nrm=normalize_case_name, side=False)
    # side_index          = build_district_tfidf_index(df, nrm=normalize_case_name, side=True)
    # run_appellate_linking(df, whole_index, side_index)
    # print("Appellate matching done")
    promoted_judges     = judges_promoted_from_district(judge_info)
    pj_stats            = compute_district_overturns(df, judge_info,
                                      api_path="batch_runs/api_responses.jsonl",
                                      mapping_path="results/appellate_matches.json")
    pj_stats.to_csv("results/promoted_judge_stats.csv", index=False)