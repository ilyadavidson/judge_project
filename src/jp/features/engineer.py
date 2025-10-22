import re

import pandas               as pd
import numpy                as np

from jp.utils.text          import norm_string, _to_list

def compute_overturns(judges: pd.DataFrame, full_data: pd.DataFrame, cutoff) -> pd.DataFrame:
    """
    For each judge, compute the overturn rate of their appealed cases. 
    
    :param judges: DataFrame with judge information, must include 'judge id' and 'promotion_date'.
    :param full_data: DataFrame with case information, must include 'district judge id', 'opinion', and 'decision_date'.
    :param cutoff: Number of months before promotion to consider cases for overturn rate calculation.
    """
    
    df = full_data.copy()
    df["judge id"]      = pd.to_numeric(df["district judge id"], errors="coerce").astype("Int64")
    df["opinion"]       = df["opinion"].astype(str).str.lower()
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

    j = judges.copy()
    j["promotion_date"] = pd.to_datetime(j["promotion_date"], errors="coerce")
    j["cutoff"]         = j["promotion_date"] - pd.DateOffset(months=cutoff)

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

def prestige_calculator(j, weight_a, weight_b, weight_c):
    """
    Compute a prestige index for each judge based on their education and career. Indicators are law school rank, clerkship experience, and law professor status.

    :param weight_*: Weights for each indicator in the prestige index calculation. For now, set as 1 as we have no prior on their relative importance. 
    Could do a logistic regression on promotion and have those coefficients be the weights. 
    """
    judges  = j.copy()

    a1      =  weight_a
    a2      =  weight_b
    a3      =  weight_c

    s       = judges[['school 1','school 2']].fillna('').agg(' '.join,axis=1).str.lower()
    d       = judges[['degree 1','degree 2']].fillna('').agg(' '.join,axis=1).str.lower()
    pc      = judges['professional career'].fillna('').str.lower()

    LSR     = (s.str.contains(r'(harvard|yale|columbia|michigan|stanford).*law') | (d.str.contains(r'j\.?d|ll\.?b|llm|s\.?j\.?d') & s.str.contains(r'harvard|yale|columbia|michigan|stanford'))).astype(int)
    CP      = pc.str.contains(r'law clerk.*(supreme court|u\.s\. supreme court|court of appeals|circuit)').astype(int)
    LP      = (pc.str.contains(r'\b(assistant|associate)?\s*professor\b') & ~pc.str.contains(r'\b(adjunct|visiting|lecturer)\b')).astype(int)

    judges['PrestigeIndex'] =   a1*LSR + a2*CP + a3*LP

    return judges['PrestigeIndex']

def _us_support_from_text(case_name: str, opinion_text: str):
    """
    Given the case name and opinion text, determine if the judge ruled in favor of the United States (1) or against it (0).
    """
    US_TXT   = re.compile(r'\b(united states|u\.?\s*s\.?a?\.?|u\.?\s*s\.?)\b', re.I)
    CRIM_CTX = re.compile(r'\b(indictment|information|prosecution|u\.?s\.? attorney|grand jury|government)\b', re.I)

    if not isinstance(opinion_text, str) or not opinion_text.strip():
        return None
    t = opinion_text.lower()

    # who is the U.S. in the caption?
    us_side = None
    if isinstance(case_name, str):
        c = case_name.lower()
        if ' v. ' in c or ' v ' in c:  # covers "v." and some OCR variants
            splitter = ' v. ' if ' v. ' in c else ' v '
            left, right = [p.strip() for p in c.split(splitter, 1)]
        elif 'vs.' in c or 'vs ' in c:
            splitter = 'vs.' if 'vs.' in c else 'vs '
            left, right = [p.strip() for p in c.split(splitter, 1)]
        else:
            left, right = c, ''
        if US_TXT.search(left):  us_side = 'plaintiff'
        if US_TXT.search(right): us_side = 'defendant'

    # fallback: criminal context mentions U.S. → assume U.S. = plaintiff
    if us_side is None and US_TXT.search(t) and CRIM_CTX.search(t):
        us_side = 'plaintiff'

    # ----- disposition cues -----
    # motions to suppress
    if re.search(r'\b(grant|granted|grants|allow|allowed|allows)\b.*\bmotion to suppress\b', t):
        return 0 if us_side in (None, 'plaintiff') else 1
    if re.search(r'\b(deny|denied|denies)\b.*\bmotion to suppress\b', t):
        return 1 if us_side in (None, 'plaintiff') else 0

    # demurrer / indictment phrasing
    if re.search(r'\bdemurrer (overruled|denied)\b', t):                     # govt wins
        return 1 if us_side != 'defendant' else 0
    if re.search(r'\bdemurrer (sustained|allowed|granted)\b', t):            # defense wins
        return 0 if us_side != 'defendant' else 1
    if re.search(r'\b(motion to (dismiss|quash).*(indictment|count)|motion in arrest of judgment)\b.*\b(denied|overruled)\b', t):
        return 1 if us_side != 'defendant' else 0
    if re.search(r'\b(motion to (dismiss|quash).*(indictment|count)|motion in arrest of judgment)\b.*\b(granted|allowed|sustained)\b', t):
        return 0 if us_side != 'defendant' else 1

    # burden-of-proof / quantity findings adverse to govt
    if re.search(r'\b(government|united states)\b.*\b(has not|hasn\'t|failed? to|did not|does not)\b.*\b(prove|establish|meet|carry)\b', t):
        return 0 if us_side in (None, 'plaintiff') else 1
    if re.search(r'\b(beyond a reasonable doubt)\b.*\b(not (proven|met)|insufficient)\b', t):
        return 0 if us_side in (None, 'plaintiff') else 1
    if re.search(r'\bmandatory minimum\b.*\b(not|no longer)\b.*\b(apply|applies|trigger|triggered)\b', t):
        return 0 if us_side in (None, 'plaintiff') else 1

    # explicit judgments
    if 'judgment for the united states' in t: return 1
    if 'judgment for the defendant'      in t: return 0

    return None

def us_support_calculator(j: pd.DataFrame, cases: pd.DataFrame,) -> pd.DataFrame:
    """
    For each judge, compute the rate at which they ruled in favor of the United States. This is done by analyzing the opinion text of each district case they authored.
    """
    c                       = cases.copy()

    c['ruled_for_US']       = c[['name', 'opinion_text']].apply(
        lambda x: _us_support_from_text(x['name'], x['opinion_text']), axis=1
    )

    c                       = c[c['opinion_author_id'].notna() & c['ruled_for_US'].notna()]

    rate                    = c.groupby('opinion_author_id')['ruled_for_US'].mean()
    N                       = c.groupby('opinion_author_id')['ruled_for_US'].size()

    j                       = j.copy()
    j['US_support_rate']    = j['judge id'].map(rate)
    j['US_support_N']       = j['judge id'].map(N).fillna(0).astype(int)

    return j['US_support_rate'], j['US_support_N']

def politicality_calculator(j, c,
                            judge_id_col='judge id',
                            case_judge_col='opinion_author_id',
                            score_col='politicality'):
    """
    For each judge, compute the average politicality score of their judged cases.
    """

    stats = (c.dropna(subset=[case_judge_col, score_col])
               .groupby(case_judge_col)[score_col]
               .agg(Politicality='mean', Politicality_N='size'))
    
    j     = j.merge(stats, left_on=judge_id_col, right_index=True, how='left')
    
    return j['Politicality']

def citation_calculator(judges,
                               cases,
                               cite_col='cite',
                               cites_to_col='cites_to',
                               judge_col='opinion_author_id',
                               judge_id_col='judge id'):
    """
    For each judge, compute a CitationImpact score based on:
        CitationImpact_i = (1 / |C_i|) * Σ log(1 + Citations_c)
    where Citations_c is the number of times case c was cited by others.
    """

    ## 1. Prepare the base data and obtain the cite identifiers
    #################################################################
    df           = cases[[cite_col, cites_to_col, judge_col]].dropna(subset=[cite_col, judge_col]).copy()
    df[cite_col] = df[cite_col].map(norm_string)

    ## 2. Build mapping: cite string to judge
    #################################################################
    cite_to_judge = df.set_index(cite_col)[judge_col].to_dict()

    ## 3. Explode cites_to into one row per citation
    #################################################################
    df[cites_to_col] = df[cites_to_col].apply(_to_list)
    x = df.explode(cites_to_col, ignore_index=True)
    x = x[x[cites_to_col].notna()] 
    x['cited_cite']     = x[cites_to_col].map(norm_string)
    x['cited_judge']    = x['cited_cite'].map(cite_to_judge)

    ## 4. Count incoming citations per case
    #################################################################
    citation_counts     = (x.groupby('cited_cite')
                         .size()
                         .rename('Citations_c'))

    ## 5. Merge back into original case data
    #################################################################
    df                  = df.merge(citation_counts, left_on=cite_col, right_index=True, how='left').fillna({'Citations_c': 0})

    ## 6. Compute CitationImpact per judge
    #################################################################
    citation_impact     = (df.groupby(judge_col)['Citations_c']
                         .apply(lambda s: np.mean(np.log1p(s)))
                         .rename('CitationImpact'))

    judges                      = judges.copy()
    judges['CitationImpact']    = judges[judge_id_col].map(citation_impact)

    return judges['CitationImpact']