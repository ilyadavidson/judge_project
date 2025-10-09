"""
Helper functions for appellate_mapping
"""

import re
import pandas as pd
import tiktoken
import re, unicodedata
import matplotlib.pyplot as plt

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

# Helpers for second-check judge/docket number
def _norm_docket(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", "", s).upper()

def _find_docket_in_text(op_text):
    _DOCKET_PAT = re.compile(r"\b[Nn]os?\.\s*([A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*)")
    if not isinstance(op_text, str) or not op_text.strip():
        return ""
    m = _DOCKET_PAT.search(op_text)
    return m.group(1).strip() if m else ""

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

def plot_distributions(df, columns):
    """
    Plot distribution(s) of selected columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing your results.
    columns : list[str]
        List of column names to plot. 
        Supported: categorical columns (bar) and 'politicality_score' (hist).
    """
    n = len(columns)
    if n == 0:
        print("No columns provided.")
        return
    
    # Determine subplot grid size
    rows = (n + 1) // 2 if n > 1 else 1
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    
    # Flatten axes into list for easy indexing
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        ax = axes[i]
        if col == "politicality_score":
            df[col].plot(
                kind="hist", bins=range(1, 7), rwidth=0.8, ax=ax, color="goldenrod"
            )
            ax.set_title("Politicality Score Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_xticks(range(1, 6))
        else:
            df[col].value_counts().plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title(f"{col.replace('_', ' ').title()} Distribution")
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
    
    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# -------- helpers --------

import re
import pandas as pd

# --- helpers ---
_pat_judge = re.compile(r"""(?ix)
    (?:^|[^\w])                                   # boundary
    (?:
        # Form A: LAST, District Judge
        (?P<a_last>[A-Z][A-Z'\-]+)                # ALLCAPS last often in captions
        \s*,\s*District\s+Judge\b
      |
        # Form B: District Judge First M. Last
        District\s+Judge\s+
        (?P<b_full>(?:Hon\.?\s+)?[A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+){0,3})
    )
""")

def _strip(s):
    if s is None:
        return ""
    try:
        # Handle pandas/NumPy NaN
        import pandas as pd
        if pd.isna(s):
            return ""
    except Exception:
        pass
    return str(s).strip()

def _first_tok(s):
    s = _strip(s)
    return s.split()[0] if s else ""

def _canon(s):
    s = _strip(s).lower()
    return re.sub(r"[^a-z]", "", s)

def _extract_court(text: str) -> str:
    m = re.search(r"United\s+States\s+District\s+Judge\s+for\s+the\s+([^,]+)", str(text or ""), flags=re.I)
    return _strip(m.group(1)) if m else ""

def _extract_judge_full(text: str) -> str:
    if not text:
        return ""
    m = _pat_judge.search(str(text))
    if not m:
        return ""
    if m.group("b_full"):
        return _strip(re.sub(r"^Hon\.?\s+", "", m.group("b_full"), flags=re.I))
    return _strip(m.group("a_last").title())  # e.g., 'KELLY' -> 'Kelly'

# --- main ---
def judge_name_to_id(
    cl_data: pd.DataFrame,
    judges_info: pd.DataFrame,
    what: str,
    text_col: str = "combined_preview"
) -> pd.Series:
    """
    Extract the district judge name/court from opinion text and resolve to judge id when possible.

    'what' in {'id','jid','judge id','last name','last','lastname','name','full name','fullname','court','court name'}
    """
    out = cl_data[[text_col]].copy()
    out["district judge_full"] = out[text_col].map(_extract_judge_full)

    # ↓ ensure safe strings before splitting/mapping
    out["district judge_full"] = out["district judge_full"].fillna("")
    out["district judge"]      = out["district judge_full"].str.split().str[-1].str.lower()
    out["last_key"]            = out["district judge"].fillna("").map(_canon)

    out["court_full"]          = out[text_col].map(_extract_court)
    out["court_key"]           = out["court_full"].fillna("").map(_canon)

    ji = judges_info.copy()
    ji["last_key"]  = ji["last name"].fillna("").map(_canon)
    ji["first_key"] = ji["first name"].fillna("").map(lambda x: _canon(_first_tok(x)))
    ji["court_key"] = ji["court name"].fillna("").map(_canon)
    ji["jid"]       = pd.to_numeric(ji["judge id"], errors="coerce").astype("Int64")

    def _resolve_id(row):
        subset = ji[ji["last_key"] == row["last_key"]]
        if len(subset) == 1:
            return subset["jid"].iloc[0]
        first, courtk = _canon(_first_tok(row["district judge_full"])), row["court_key"]
        s2 = subset[subset["first_key"] == first]
        if len(s2) == 1:
            return s2["jid"].iloc[0]
        if len(s2) > 1 and courtk:
            s3 = s2[s2["court_key"].apply(lambda x: bool(x) and (courtk in x or x in courtk))]
            if len(s3):
                return s3["jid"].iloc[0]
        return pd.NA

    key = what.strip().lower()
    if key in {"id","jid","judge id"}:
        return out.apply(_resolve_id, axis=1).astype("Int64").reindex(cl_data.index)
    if key in {"last name","last","lastname"}:
        return out["district judge"].reindex(cl_data.index)
    if key in {"name","full name","fullname"}:
        return out["district judge_full"].reindex(cl_data.index)
    if key in {"court","court name"}:
        return out["court_full"].reindex(cl_data.index)
    raise ValueError("what must be one of: 'id', 'last name', 'name', 'court'.")

import re, pandas as pd, unicodedata

def extract_district_judge_info(cl_data: pd.DataFrame, judges_info: pd.DataFrame) -> pd.DataFrame:
    """Add 'district judge' (last name, lowercase) and 'district judge id' (Int64) to cl_data."""
    # --- helpers ---
    def _norm(s):  # normalize for matching
        return unicodedata.normalize("NFKD", str(s or "")).strip()

    def _canon(s):  # canonical key: lowercase letters only
        return re.sub(r"[^a-z]", "", _norm(s).lower())

    def _strip_honorifics(s):
        s = re.sub(r"(?i)\b(the\s+)?honorable\b|^hon\.?\s*", "", _norm(s))
        s = re.sub(r"\s+", " ", s).strip(" ,;")
        return s

    SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
    def _split_name(full):
        # keep only word-ish tokens; drop suffixes at end
        toks = re.findall(r"[A-Za-z][A-Za-z'\.-]*", _strip_honorifics(full))
        while toks and toks[-1].rstrip(".").lower() in SUFFIX:
            toks.pop()
        if not toks: 
            return "", ""
        first = re.sub(r"\.$", "", toks[0])
        last  = re.sub(r"\.$", "", toks[-1])
        return first, last

    # patterns (non-greedy, stop at line end; forbid digits in names)
    # 1) "District Judge: Hon. Jennifer P. Wilson"
    pat_after = re.compile(
        r"(?im)\bdistrict\s+judge\b[:\s,]*"
        r"(?:(?:the\s+)?honorable|hon\.)?\s*"
        r"(?P<name>[A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*$"
    )
    # 2) "SCIRICA, District Judge."
    pat_before = re.compile(
        r"(?im)(?P<name>[A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:U\.S\.\s*)?district\s+judge\b"
    )
    # 3) looser single-line fallback near "District Judge"
    pat_line = re.compile(
        r"(?im)^(?P<line>.*\bdistrict\s+judge\b.*)$"
    )
    # court: "United States District Court for the Middle District of Pennsylvania"
    pat_court = re.compile(
        r"(?im)(?:on\s+)?appeal\s+from\s+the\s+united\s+states\s+district\s+court\s+for\s+the\s+(?P<court>[^\r\n,]+)"
    )

    def _extract_name(txt: str) -> str:
        if not txt: return ""
        # prefer exact one-line “after” form
        for m in pat_after.finditer(txt):
            return _strip_honorifics(m.group("name"))
        # then “before” form
        for m in pat_before.finditer(txt):
            return _strip_honorifics(m.group("name"))
        # fallback: scan the first line that contains "district judge" and try to pick a clean name nearby
        m = pat_line.search(txt)
        if m:
            line = m.group("line")
            # try ": <name>" after the phrase
            m2 = re.search(
                r"(?i)\bdistrict\s+judge\b[:\s,]*"
                r"(?:(?:the\s+)?honorable|hon\.)?\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])", line)
            if m2: return _strip_honorifics(m2.group(1))
            # or "<NAME>, District Judge"
            m3 = re.search(
                r"([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:U\.S\.\s*)?district\s+judge\b", line, flags=re.I)
            if m3: return _strip_honorifics(m3.group(1))
        return ""

    def _extract_court(txt: str) -> str:
        if not txt: return ""
        m = pat_court.search(txt)
        if m: return _norm(m.group("court"))
        # softer fallback if phrasing differs slightly
        m2 = re.search(r"(?im)united\s+states\s+district\s+court\s+for\s+the\s+([^\r\n,]+)", txt)
        return _norm(m2.group(1)) if m2 else ""

    # prep judge reference (canonical keys)
    J = judges_info.copy()
    J["jid"]       = pd.to_numeric(J["judge id"], errors="coerce").astype("Int64")
    J["last_key"]  = J["last name"].map(_canon)
    J["first_key"] = J["first name"].map(lambda x: _canon(re.sub(r"\.$","", (str(x).split() or [""])[0])))
    J["court_key"] = J["court name"].map(_canon)

    def _resolve_id(full_name: str, court_str: str) -> pd.Series:
        first, last = _split_name(full_name)
        last_key  = _canon(last)
        first_key = _canon(first)
        court_key = _canon(court_str)
        if not last_key:
            return pd.Series({"district judge": "", "district judge id": pd.NA})

        cand = J[J["last_key"] == last_key]
        if len(cand) == 1:
            return pd.Series({"district judge": last.lower(), "district judge id": cand["jid"].iloc[0]})
        if len(cand) > 1 and first_key:
            cand2 = cand[cand["first_key"] == first_key]
            if len(cand2) == 1:
                return pd.Series({"district judge": last.lower(), "district judge id": cand2["jid"].iloc[0]})
            if len(cand2) > 1 and court_key:
                c3 = cand2[cand2["court_key"].apply(lambda x: bool(x) and (court_key in x or x in court_key))]
                if len(c3):
                    return pd.Series({"district judge": last.lower(), "district judge id": c3["jid"].iloc[0]})
        # if still ambiguous or no match
        return pd.Series({"district judge": last.lower(), "district judge id": pd.NA})

    # apply per row (robustness > minimal lines)
    out = cl_data.copy()
    texts = out["opinion_text"].astype(str)
    names = texts.map(_extract_name)
    courts = texts.map(_extract_court)
    resolved = [ _resolve_id(n, c) for n, c in zip(names, courts) ]
    resolved_df = pd.DataFrame(resolved, index=out.index)
    out["district judge"] = resolved_df["district judge"]
    out["district judge id"] = resolved_df["district judge id"].astype("Int64")
    return out