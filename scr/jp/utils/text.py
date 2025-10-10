import re
import pandas as pd

def normalize_case_name(name: str) -> str:
    """
    Normalize a case caption for fuzzy matching.
    Steps:
      1. Lowercase
      2. Standardize/remove 'v.' / 'vs.' -> remove connector
      3. Drop punctuation
      4. Remove stopwords (case-insensitive)
    """

    _STOPWORDS = {
        "appellant","appellee","petitioner","respondent","defendant","plaintiff","appeal",
        "intervenor","movant","united","states","of","america","the","et","al","and","a","an",
        "as","ex","rel","for","its","etal","department","dept","district","commonwealth",
        "people","state","in","re","debtor","business","appellants","enterprise","hereafter",
        "hereinafter","successor","trustee","trustees","plaintiffs","defendants","partners",
        "partnership","partnerships","associates","association","associations","et. al","others",
        "incorporated","incorporated.", "in", "the", "matter", "of", "and", "of the",
}
    
    _REPLACEMENTS = {
    r"\bh/?w\b": "spouse",             # h/w → spouse
    r"\bhis wife\b": "spouse",
    r"\bhusband\b": "spouse",
    r"\bwife\b": "spouse",
    r"\bcompany\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bco\b": "corp",
    r"\binc\.?\b": "corp",
    r"\bltd\.?\b": "corp",
    r"\bllc\.?\b": "corp",
    r"&": "and"
}
    
    if not isinstance(name, str) or not name.strip():
        return ""

    # lowercase
    x = name.lower()

    # standardize/remove "v.", "vs", "vs.", "v"
    x = re.sub(r"\b(vs?\.?)\b", " ", x)

    for pat, repl in _REPLACEMENTS.items():
        x = re.sub(pat, repl, x)
        
    # drop punctuation (anything not alphanumeric or whitespace)
    x = re.sub(r"[^\w\s]", " ", x)

    # split, drop stopwords, rejoin
    tokens = [t for t in x.split() if t and t not in _STOPWORDS]

    return " ".join(tokens)


def norm_id(x):
    if pd.isna(x): return None
    s = str(x).strip()
    # drop trailing ".0" from floats
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except Exception:
            return s
    return s