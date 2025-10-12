"""
This file extracts all the judge info from the scraped data and return the dataset ready to be used.
"""


import re, unicodedata
import pandas as pd
import os
from scr.jp.cl.scrape import scrape_third_circuit 

def extract_district_judge_info(cl_data: pd.DataFrame, judges_info: pd.DataFrame) -> pd.DataFrame:
    """Add 'district judge' (last name, lowercase) and 'district judge id' (Int64) to cl_data.
       Priority: parse judge near the 'Appeal from ...' block; else use generic District Judge patterns.
    """
    # ---------- helpers ----------
    def _norm(s): return unicodedata.normalize("NFKD", str(s or "")).replace("\u00A0"," ").strip()
    def _canon(s): return re.sub(r"[^a-z]", "", _norm(s).lower())
    def _strip_honorifics(s):
        s = re.sub(r"(?i)\b(the\s+)?honorable\b|^hon\.?\s*", "", _norm(s))
        return re.sub(r"\s+", " ", s).strip(" ,;")
    SUFFIX = {"jr","sr","ii","iii","iv","v"}
    def _split_name(full):
        toks = re.findall(r"[A-Za-z][A-Za-z'\.-]*", _strip_honorifics(full))
        while toks and toks[-1].rstrip(".").lower() in SUFFIX: toks.pop()
        if not toks: return "",""
        first = re.sub(r"\.$","", toks[0]); last = re.sub(r"\.$","", toks[-1])
        return first, last

    # court extraction (full and abbrev)
    pat_court_full  = re.compile(r"(?im)(?:on\s+)?appeal\s+from\s+the\s+united\s+states\s+district\s+court\s+for\s+the\s+([^\r\n,]+)")
    pat_court_short = re.compile(r"(?im)appeal\s+from[:\s,]*([A-Z]\.[A-Z]\.[A-Za-z]+|[A-Z]\.[A-Z]\.[A-Z]\.|[A-Z]\.[A-Z]\.[A-Za-z]+\.[A-Za-z]+|[A-Z]\.[A-Za-z]+\.[A-Za-z]+)")  # e.g., D.N.J., E.D. Pa., W.D.Pa.

    # “District Judge …” (after), “…, District Judge” (before), and line fallback
    pat_after  = re.compile(r"(?im)\bdistrict\s+judge\b[:\s,]*?(?:(?:the\s+)?honorable|hon\.)?\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])")
    pat_before = re.compile(r"(?im)([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:U\.S\.\s*)?district\s+judge\b")
    pat_linedj = re.compile(r"(?im)^.*\bdistrict\s+judge\b.*$")

    # near-appeal name patterns (parenthetical, “X, J.” / “X, D.J.”, inline “District Judge: …”)
    pat_paren  = re.compile(r"(?im)\(\s*(?:the\s+)?honorable|hon\.\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*\)")
    pat_commaJ = re.compile(r"(?im)^\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:D\.?J\.?|J\.)\s*$")
    pat_inline = re.compile(r"(?im)\bdistrict\s+judge\b[:\s,]*?(?:(?:the\s+)?honorable|hon\.)?\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])")

    # Judges reference with canonical keys
    J = judges_info.copy()
    J["jid"]       = pd.to_numeric(J["judge id"], errors="coerce").astype("Int64")
    J["last_key"]  = J["last name"].map(_canon)
    J["first_key"] = J["first name"].map(lambda x: _canon(re.sub(r"\.$","", (str(x).split() or [""])[0])))
    J["court_key"] = J["court name"].map(_canon)

    def _resolve_id(full_name: str, court_str: str):
        first,last = _split_name(full_name)
        last_key, first_key, court_key = _canon(last), _canon(first), _canon(court_str)
        if not last_key: return "", pd.NA
        cand = J[J["last_key"]==last_key]
        if len(cand)==1: return last.lower(), cand["jid"].iloc[0]
        if len(cand)>1 and first_key:
            c2 = cand[cand["first_key"]==first_key]
            if len(c2)==1: return last.lower(), c2["jid"].iloc[0]
            if len(c2)>1 and court_key:
                c3 = c2[c2["court_key"].apply(lambda x: bool(x) and (court_key in x or x in court_key))]
                if len(c3): return last.lower(), c3["jid"].iloc[0]
        if len(cand)>1 and court_key:
            c4 = cand[cand["court_key"].apply(lambda x: bool(x) and (court_key in x or x in court_key))]
            if len(c4)==1: return last.lower(), c4["jid"].iloc[0]
        return last.lower(), pd.NA

    def _extract_from_appeal_block(txt: str):
        if not txt: return "",""
        lines = _norm(txt).splitlines()
        # locate the first “appeal from …” line
        idx = next((i for i,l in enumerate(lines) if re.search(r"(?i)\bappeal\s+from\b", l)), None)
        if idx is None:  # also try “On Appeal from …”
            idx = next((i for i,l in enumerate(lines) if re.search(r"(?i)\bon\s+appeal\s+from\b", l)), None)
        if idx is None: return "",""
        block = lines[idx: min(len(lines), idx+10)]  # look ahead a few lines

        # court (prefer full)
        m = pat_court_full.search("\n".join(block)) or pat_court_full.search(txt)
        court = m.group(1).strip() if m else ""
        if not court:
            m2 = pat_court_short.search(lines[idx])
            court = m2.group(1).strip() if m2 else ""

        # 1) parenthetical "(Honorable …)"
        for l in block:
            mp = re.search(r"(?i)\((?:\s*(?:the\s+)?honorable|hon\.)\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*\)", l)
            if mp: return _strip_honorifics(mp.group(1)), court

        # 2) a bare line like "Mencer, J." / "Fullam, D.J."
        for l in block:
            mj = pat_commaJ.match(l.strip())
            if mj: return _strip_honorifics(mj.group(1)), court

        # 3) inline “District Judge: …” within block
        for l in block:
            mi = pat_inline.search(l)
            if mi: return _strip_honorifics(mi.group(1)), court

        # 4) any “District Judge …” within the block line
        for l in block:
            ma = pat_after.search(l)
            if ma: return _strip_honorifics(ma.group(1)), court
            mb = pat_before.search(l)
            if mb: return _strip_honorifics(mb.group(1)), court

        return "",""

    def _extract_fallback(txt: str):
        if not txt: return "",""
        # Prefer precise single-line “after” then “before”
        ma = pat_after.search(txt)
        if ma: return _strip_honorifics(ma.group(1)), ""
        mb = pat_before.search(txt)
        if mb: return _strip_honorifics(mb.group(1)), ""
        # scan first line mentioning “district judge” and try inline patterns
        mline = pat_linedj.search(txt)
        if mline:
            line = mline.group(0)
            mi = pat_inline.search(line)
            if mi: return _strip_honorifics(mi.group(1)), ""
            mj = pat_before.search(line)
            if mj: return _strip_honorifics(mj.group(1)), ""
        return "",""

    def _ascii(s: str) -> str:
        return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")

    # ---------- apply ----------
    out = cl_data.copy()
    texts = out["opinion_text"].astype(str)

    names, courts = [], []
    for t in texts:
        t = _ascii(t)
        n,c = _extract_from_appeal_block(t)
        if not n: n,c = _extract_fallback(t)
        names.append(n); courts.append(c)

    # resolve to ids
    resolved = [ _resolve_id(n,c) for n,c in zip(names, courts) ]
    rdf = pd.DataFrame(resolved, columns=["district judge","district judge id"], index=out.index)
    out["district judge"]    = rdf["district judge"]
    out["district judge id"] = rdf["district judge id"].astype("Int64")
    return out


def cl_loader(cl_data, judges):
    thrd_judges = judges[judges['court name'].str.contains(r"Third|Delaware|New Jersey|Pennsylvania|Virgin Islands")]
 
    PHRASE = r"(?i)(?<!\w)(?:on[\s\u00A0]+)?appeal[\s\u00A0]+from(?!\w)|(?<!\w)district[\s\u00A0]+judge(?!\w)"
    cl_clean = cl_data[cl_data["opinion_text"].str.contains(PHRASE, case=False, regex=True, na=False)].reset_index(drop=True)
    cl_extracted = extract_district_judge_info(cl_clean, judges)

    cl_clean = cl_extracted[cl_extracted['district judge id'].notna()]
    cl_clean = cl_extracted[cl_extracted["district judge id"].isin(thrd_judges["judge id"].dropna().astype(int))]
    cl_clean['is_appellate'] = 1
    cl_clean['unique_id'] = 'CL_' + cl_clean.index.astype(str)
    cl_clean = cl_clean[cl_clean['district judge id'].notna()]

    cl_clean = cl_clean[cl_clean["opinion_text"].notna()]

    return cl_clean