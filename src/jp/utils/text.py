import re
import pandas as pd
import tiktoken
from pathlib                 import Path

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

def split_on_v(name: str):
    """
    Split caption into (left, right) around the **rightmost** legal connector.
    Avoids splitting on middle initials like 'Johnny V. Brown'.
    Returns (None, None) if no valid split. Outputs are lowercased & trimmed.
    """

    # Looks for any variation of v./vs./versus
    ############################################################################################################
    _V_CONN = re.compile(r"\s+(?:v(?:\.|s\.?)?|versus)\s+", re.IGNORECASE)

    if not isinstance(name, str) or not name.strip():
        return (None, None)

    s       = " ".join(name.strip().split())  # collapse weird whitespace
    last    = None
    
    # If there's a clear defendant and plaintiff side, split these groups. If not, keep them the same. 
    ############################################################################################################
    for m in _V_CONN.finditer(s):
        last = m
    if not last:
        return (None, None)

    left    = s[:last.start()].strip().lower()
    right   = s[last.end():].strip().lower()

    if left and right:
        return left, right
    return (None, None)

def split_normalize_dockets(s):
    # takes the docket number out 
    if not isinstance(s, str) or not s.strip():
        return []
    x = re.sub(r"\b[Nn]os?\.\s*", "", s)
    parts = re.split(r"[,\;/\s]+", x)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = p.replace("–","-").replace("—","-")
        p = re.sub(r"[^A-Za-z0-9\-/\.]+", "", p)
        if p:
            out.append(p.upper())
    return out

def extract_all_dockets(text):
    _DOCKET_PAT = re.compile(r"\b[Nn]os?\.\s*[A-Za-z0-9\-_/\. ,;]+")
    if not isinstance(text, str) or not text.strip():
        return []
    m = _DOCKET_PAT.search(text)  # first “No.”/“Nos.” block near the top
    if not m:
        return []
    return split_normalize_dockets(m.group(0))


def _candidate_judge_names(raw):
    """
    Extract plausible judge-name needles from opinion_author_raw.
    - split on ; , and 'and'
    - keep alphabetic tokens
    - drop titles + suffixes (Jr, Sr, II, III, IV, ...)
    - return last names and first+last combos
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    # normalize separators
    x              = re.sub(r"\band\b", ",", raw, flags=re.I)
    parts          = re.split(r"[;,\n]+", x)

    needles        = set()
    drop_titles    = {"judge","chief","circuit","district","magistrate","senior","acting","visiting","bankruptcy"}
    drop_suffixes  = {"jr","jr.","sr","sr.","ii","iii","iv","v"}

    # Roman numeral pattern
    roman_pat = re.compile(
        r"^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})"
        r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I
    )

    for part in parts:
        toks        = [t for t in re.findall(r"[A-Za-z]+", part) if t]
        toks_lower  = [t.lower() for t in toks]

        # drop titles, suffixes, and roman numerals
        toks = [
            t for t, l in zip(toks, toks_lower)
            if l not in drop_titles and l not in drop_suffixes and not roman_pat.match(t)
        ]

        if not toks:
            continue

        # last-name candidate
        needles.add(toks[-1])

        # first+last combo
        if len(toks) >= 2:
            needles.add(f"{toks[0]} {toks[-1]}")

    return list(needles)

def _text_contains_any(hay: str, needles: list[str]) -> bool:
    if not isinstance(hay, str) or not hay.strip() or not needles:
        return False
    
    H = hay.lower()
    for n in needles:
        n = n.strip()
        if not n:
            continue
        # whole-word for single token; substring for two-token form
        if " " in n:
            if n.lower() in H:
                return True
        else:
            if re.search(rf"\b{re.escape(n.lower())}\b", H):
                return True
    return False

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

def _canon(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s or "").lower())

def _infer_circuit(app_court_name: str) -> str | None:
    """
    Return a normalized circuit key (e.g., 'first','second','third','fourth',...,'dc','federal')
    from an appellate 'court name' string. Returns None if not identified.
    """
    s = _canon(app_court_name)
    # common signals
    for key in ["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","eleventh"]:
        if key in s and ("circuit" in s or "ctappeals" in s or "courtofappeals" in s):
            return key
    return None

def truncate_opinion(text, max_tokens= 6000) -> str:
    enc = tiktoken.get_encoding("o200k_base")
    text = "" if text is None else str(text)
    toks = enc.encode(text)
    if len(toks) > max_tokens:
        head = toks[:max_tokens]
        tail = toks[-max_tokens:]
        toks = head + tail
    return enc.decode(toks)

def text_builder(df, limit, mx_tk):
    """ 
    Function to call in dataset.
    
    :param df: original df.
    :param limit: how many cases to load.
    :param max_tokens_each: how many of the last tokens we want to keep.
    """
    if limit    == None:
        subset  = df[df['is_appellate']==1].copy()
    else:
        subset  = df[df['is_appellate']==1].head(limit).copy()
    
    results = []

    for _, row in subset.iterrows():
        cid         = row["unique_id"]
        raw_text    = row["opinion_text"]
        trimmed     = truncate_opinion(raw_text, max_tokens=mx_tk)
        results.append({"id": cid, "text": trimmed})

    return results

def ensure_dir(*parts) -> Path:
    path = Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

def _norm_circuit(c) -> str:
    """
    Given a circuit name like '4th', '2nd', or '3d',
    return only the corresponding CourtListener code 'ca4', 'ca2', etc.
    Assumes user input is valid and properly formatted.
    """
    s = str(c).strip().lower()
    num = "".join(ch for ch in s if ch.isdigit())
    if not num:
        raise ValueError(f"Expected a numbered circuit like '4th' or '3d', got '{c}'")
    return f"ca{num}"

def judge(name: str):
    judges = pd.read_csv('data/judge_info.csv')
    judge = judges[judges['last name']==name]
    return judge

def norm_string(s): return re.sub(r'\s+', ' ', str(s).strip().lower())

def _to_list(v):
    if isinstance(v, list): return v
    if pd.isna(v): return []
    if isinstance(v, str):
        s = v.strip()
        # if it looks like a JSON-ish list, parse it
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try: return list(ast.literal_eval(s))
            except: pass
        # else split on commas
        return [t.strip() for t in s.split(',') if t.strip()]
    return [str(v)]