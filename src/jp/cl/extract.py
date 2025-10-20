"""
This file extracts all the judge info from the scraped data and return the dataset ready to be used.

TO-DO: filter for judge being in the appropriate court.
"""
import re, unicodedata
import pandas       as pd

def extract_district_judge_info(cl_data: pd.DataFrame, judges_info: pd.DataFrame, cid) -> pd.DataFrame:
    """
    Add 'district judge' (last name, lowercase) and 'district judge id' (Int64) to cl_data,
    using circuit-specific extraction patterns determined by `cid` (e.g., '4th', '2nd', '9th').
    Court name is still parsed by the existing patterns and used for ID disambiguation.
    """

    # ---------- helpers ----------
    _norm = lambda s: unicodedata.normalize("NFKD", str(s or "")).replace("\u00A0", " ").strip()
    _canon = lambda s: re.sub(r"[^a-z]", "", _norm(s).lower())

    def _strip_honorifics(s: str) -> str:
        s = re.sub(r"(?i)\b(?:the\s+)?honorable\b|^hon\.?\s*", "", _norm(s))
        return re.sub(r"\s+", " ", s).strip(" ,;.")

    SUFFIX = {"jr","sr","ii","iii","iv","v"}
    def _split_name(full: str):
        toks = re.findall(r"[A-Za-z][A-Za-z'\.-]*", _strip_honorifics(full))
        while toks and toks[-1].rstrip(".").lower() in SUFFIX: toks.pop()
        if not toks: return "",""
        first = re.sub(r"\.$","", toks[0]); last = re.sub(r"\.$","", toks[-1])
        return first, last

    def _ascii(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "")
        s = s.replace("–","-").replace("—","-").replace("\u00A0"," ")
        return s.encode("ascii","ignore").decode("ascii")

    # court extraction (full and abbrev)
    pat_court_full  = re.compile(r"(?im)(?:on\s+)?appeal\s+from\s+the\s+united\s+states\s+district\s+court\s+for\s+the\s+([^\r\n,]+)")
    pat_court_short = re.compile(r"(?im)appeal\s+from[:\s,]*([A-Z]\.[A-Z]\.[A-Za-z]+|[A-Z]\.[A-Z]\.[A-Z]\.|[A-Z]\.[A-Z]\.[A-Za-z]+\.[A-Za-z]+|[A-Z]\.[A-Za-z]+\.[A-Za-z]+)")

    # generic patterns (used as fallbacks/for shared pieces)
    pat_after   = re.compile(r"(?im)\bdistrict\s+judge\b[:\s,]*?(?:(?:the\s+)?honorable|hon\.)?\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])")
    pat_before  = re.compile(r"(?im)([A-Z][A-Za-z'\.\- ]*[A-Za-z](?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?)\s*,\s*(?:U\.S\.\s*)?(?:Chief\s+|Senior\s+)?district\s+judge\b")
    pat_inline  = re.compile(
        r"(?im)(?:"
        r"\b(?:U\.?\s*S\.?\s*)?(?:(?:Chief|Senior)\s+)?(?:Magistrate\s+Judge|District\s+Judge|Judge)\b"
        r"[:\s,]*?(?:(?:the\s+)?honorable|hon\.)?\s*"
        r"([A-Z][A-Za-z'\.\- ]*[A-Za-z](?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?)"     # group(1): title → name
        r"|"
        r"([A-Z][A-Za-z'\.\- ]*[A-Za-z](?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?)"     # group(2): name → title
        r"\s*,?\s*(?:-|—)?\s*(?:(?:Chief|Senior)\s+)?(?:U\.?\s*S\.?\s*)?(?:Magistrate\s+Judge|District\s+Judge|Judge)\b"
        r")"
    )
    pat_commaJ  = re.compile(r"(?im)^\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:D\.?J\.?|J\.)\s*$")
    pat_linedj  = re.compile(r"(?im)^.*\bdistrict\s+judge\b.*$")

    # ---- circuit-specific patterns (first match wins; capture group 1 contains the name) ----
    circuit_num = (re.search(r"\d+", str(cid)) or re.match(r"", "")).group(0)  # "4" for "4th"
    PAT_BY_CIRCUIT = {
        "1": [  # [Hon. William G. Young, U.S. District Judge]
            re.compile(r"(?im)\[\s*(?:Hon\.?|Honorable)\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z](?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?)\s*,\s*U\.?\s*S\.?\s*District\s*Judge\s*\]"),
        ],
        "2": [  # No. 15-cv-..., Edgardo Ramos, Judge.
            re.compile(r"(?im)\bNo(?:s)?\.\s*[\w:;\- ]+[,-]\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z](?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?)\s*,\s*Judge\."),
        ],
        "3": [
            re.compile(
                r"(?im)\b(?:Chief|Senior\s+)?District\s+Judge:\s*(?:Hon\.?|Honorable)?\s*"
                r"([A-Z][A-Za-z'\.\- ]*[A-Za-z](?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?)"
            ),
        ],
        "4": [
    # at Charlotte. Max O. Cogburn, Jr., District Judge.
    # at Alexandria. Rossie David Alston, Jr., District Judge.
    # at Asheville. William G. Young, Senior District Judge for the United States District Court
        re.compile(
            r"(?im)\bat\s+\s*[A-Za-z\.\- ]+\.\s*"
            r"([A-Z][A-Za-z'\.\- ]*[A-Za-z]"
            r"(?:,\s*(?:Jr|Sr|II|III|IV|V)\.?)?)"                        # optional suffix like ", Jr."
            r"\s*,\s*(?:Chief|Senior)?\s*District\s+Judge"               # allow Chief/Senior
            r"(?:\s+for\s+the\s+United\s+States(?:\s+District\s+Court)?)?"# optional trailing clause
            r"\.?"                                                       # optional period
        ),
        # Slightly more permissive backup in case punctuation/spacing varies
        re.compile(
            r"(?im)\bat\s+[^\S\r\n]*[A-Za-z\.\- ]+\.\s*"
            r"([A-Z][A-Za-z'\.\- ]*[A-Za-z]"
            r"(?:,\s*(?:Jr|Sr|II|III|IV|V)\.?)?)"
            r"\s*,\s*(?:Chief|Senior)?\s*District\s+Judge"
            r"(?:\s+for\s+the\s+United\s+States(?:\s+District\s+Court)?)?"
            r"\.?"
        ),
    ],
        "6": [  # No. ... — Jane M. Beckering, District Judge.
            re.compile(r"(?im)\bNo(?:s)?\.\s*[\w:;\-]+(?:\s*;\s*[\w:;\-]+)?\s*[-—]\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z](?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?)\s*,\s*District\s+Judge\."),
        ],
        "7": [  # — Damon R. Leichty, Judge. / — Nancy Joseph, Magistrate Judge.
            re.compile(r"(?im)[-—]\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:Magistrate\s+)?Judge\."),
        ],
        "9": [  # Robert J. Bryan, District Judge, Presiding
            re.compile(r"(?im)\bAppeal\s+from\s+the\s+United\s+States\s+District\s+Court.*?\n?\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*District\s+Judge,\s*Presiding"),
        ],
    }

    # Judges reference table
    J = judges_info.copy()
    J["jid"]       = pd.to_numeric(J["judge id"], errors="coerce").astype("Int64")
    J["last_key"]  = J["last name"].map(_canon)
    J["first_key"] = J["first name"].map(lambda x: _canon(re.sub(r"\.$","", (str(x).split() or [""])[0])))
    J["court_key"] = J["court name"].map(_canon)

    def _resolve_id(full_name: str, court_str: str):
        first,last = _split_name(full_name)
        last_key, first_key, court_key = _canon(last), _canon(first), _canon(court_str)
        if not last_key: return "", pd.NA
        cand = J[J["last_key"] == last_key]
        if len(cand) == 1: return last.lower(), cand["jid"].iloc[0]
        if len(cand) > 1 and first_key:
            c2 = cand[cand["first_key"] == first_key]
            if len(c2) == 1: return last.lower(), c2["jid"].iloc[0]
            if len(c2) > 1 and court_key:
                c3 = c2[c2["court_key"].apply(lambda x: bool(x) and (court_key in x or x in court_key))]
                if len(c3): return last.lower(), c3["jid"].iloc[0]
        if len(cand) > 1 and court_key:
            c4 = cand[cand["court_key"].apply(lambda x: bool(x) and (court_key in x or x in court_key))]
            if len(c4) == 1: return last.lower(), c4["jid"].iloc[0]
        return last.lower(), pd.NA

    def _extract_court(txt: str) -> str:
        m = pat_court_full.search(txt) or pat_court_full.search(_ascii(txt))
        if m: return m.group(1).strip()
        m2 = pat_court_short.search(txt)
        return m2.group(1).strip() if m2 else ""

    def _extract_name_by_circuit(txt: str) -> str:
        T = _ascii(txt)
        # try circuit-specific patterns first (if known)
        if circuit_num in PAT_BY_CIRCUIT:
            for pat in PAT_BY_CIRCUIT[circuit_num]:
                m = pat.search(T)
                if m:
                    return _strip_honorifics(m.group(1))

        # otherwise fall back to robust generic shapes
        m = pat_inline.search(T)
        if m:
            return _strip_honorifics(m.group(1) or m.group(2))
        m = pat_after.search(T)
        if m: return _strip_honorifics(m.group(1))
        m = pat_before.search(T)
        if m: return _strip_honorifics(m.group(1))

        # try a quick scan line containing 'district judge' + inline parse
        ml = pat_linedj.search(T)
        if ml:
            line = ml.group(0)
            mi = pat_inline.search(line)
            if mi: return _strip_honorifics(mi.group(1) or mi.group(2))
            mb = pat_before.search(line)
            if mb: return _strip_honorifics(mb.group(1))

        # super last-ditch: a bare "X, D.J." line
        for l in _ascii(txt).splitlines():
            mj = pat_commaJ.match(l.strip())
            if mj: return _strip_honorifics(mj.group(1))

        return ""

    # ---------- apply ----------
    out = cl_data.copy()
    texts = out["opinion_text"].astype(str)

    names, courts = [], []
    for t in texts:
        name  = _extract_name_by_circuit(t)
        court = _extract_court(t)
        names.append(name); courts.append(court)

    # resolve to ids
    resolved = [_resolve_id(n, c) for n, c in zip(names, courts)]
    rdf = pd.DataFrame(resolved, columns=["district judge","district judge id"], index=out.index)
    out["district judge"]    = rdf["district judge"]
    out["district judge id"] = rdf["district judge id"].astype("Int64")

    # keep only resolved
    out = out[out["district judge id"].notna()]
    return out

def cl_loader(cl_data, judges, cid):
    """
    Given raw CL data and judges info, return cleaned CL data with district judge info based on if the terms "Appeal from" or "District Judge" are found in the opinion text.
    """
 
    PHRASE = r"(?i)(?<!\w)(?:on[\s\u00A0]+)?appeal[\s\u00A0]+from(?!\w)|(?<!\w)district[\s\u00A0]+judge(?!\w)"

    cl_regex_hits               = cl_data[cl_data["opinion_text"].str.contains(PHRASE, case=False, regex=True, na=False)].reset_index(drop=True)
    cl_clean                    = extract_district_judge_info(cl_regex_hits, judges, cid)

    cl_clean['is_appellate']    = 1 # all are appellate cases, but keep for consistency with CAP data
    cl_clean['unique_id']       = 'CL_' + cl_clean.index.astype(str)

    return cl_clean