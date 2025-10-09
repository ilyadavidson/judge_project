import os, io, re, time, random, requests, pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tempfile import NamedTemporaryFile  

judges = pd.read_csv('data/judge_info.csv')
thrd_judges = judges[judges['court name'].str.contains(r"Third|Delaware|New Jersey|Pennsylvania|Virgin Islands")]

load_dotenv()
BASE_SEARCH = "https://www.courtlistener.com/api/rest/v4/search/"
CLUSTER_URL = "https://www.courtlistener.com/api/rest/v4/clusters/{id}/"
session = requests.Session()
session.headers.update({
    "Authorization": f"Token {os.getenv('COURTLISTENER_TOKEN')}",
    "User-Agent": os.getenv("COURTLISTENER_USER_AGENT","courtlistener-scraper/1.0")
})
RETRY = {429,500,502,503,504}

def _get_json(url, params=None, timeout=60, attempts=6):
    delay=0.7
    for a in range(attempts):
        r = session.get(url, params=params if a==0 else None, timeout=timeout)
        if r.status_code in RETRY:
            if r.status_code==429 and (ra:=r.headers.get("Retry-After")):
                try: time.sleep(float(ra))
                except: pass
            time.sleep(min(delay*(2**a)+random.uniform(0,0.4), 18))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

def _html_to_text(h):
    soup=BeautifulSoup(h or "","html.parser")
    for b in soup(["script","style"]): b.decompose()
    return re.sub(r"\n{3,}","\n\n", soup.get_text("\n")).strip()

def _pdf_to_text(b):
    reader=PdfReader(io.BytesIO(b))
    return "\n\f\n".join((p.extract_text() or "") for p in reader.pages)

def _atomic_append_csv(path, df_chunk, header_needed):
    """Write to temporary file then atomically append — avoids corruption if crash mid-write."""
    mode = "a"
    tmpf = NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path) or ".", prefix=".tmp_", suffix=".csv")
    try:
        df_chunk.to_csv(tmpf.name, index=False, header=header_needed)
        tmpf.close()
        with open(tmpf.name, "r", encoding="utf-8") as src, open(path, mode, encoding="utf-8") as dest:
            for line in src:
                dest.write(line)
    finally:
        try: os.remove(tmpf.name)
        except FileNotFoundError: pass

def _cluster_text(cid):
    cl = _get_json(
        CLUSTER_URL.format(id=cid),
        params={"fields":"plain_text,plain_text_with_citations,sub_opinions,opinions,download_url,docket,case_name,case_name_full,date_filed"}
    )
    if cl.get("plain_text"): return cl["plain_text"]
    if cl.get("plain_text_with_citations"): return cl["plain_text_with_citations"]
    subs = cl.get("sub_opinions") or cl.get("opinions") or []
    if subs:
        op_uri = subs[0] if isinstance(subs[0], str) else (subs[0].get("resource_uri") or subs[0].get("id"))
        if isinstance(op_uri, str):
            op = _get_json(op_uri, params={"fields":"plain_text,html,html_with_citations,download_url"})
        else:
            op = _get_json(f"https://www.courtlistener.com/api/rest/v4/opinions/{op_uri}/",
                           params={"fields":"plain_text,html,html_with_citations,download_url"})
        if op.get("plain_text"): return op["plain_text"]
        if op.get("html_with_citations"): return _html_to_text(op["html_with_citations"])
        if op.get("html"): return _html_to_text(op["html"])
        if (u:=op.get("download_url")):
            pr=session.get(u,timeout=120)
            if pr.ok: return _pdf_to_text(pr.content)
    if (u:=cl.get("download_url")):
        pr=session.get(u,timeout=120)
        if pr.ok: return _pdf_to_text(pr.content)
    return ""

def _robust_cluster_id(it):
    return (it.get("cluster_id")
            or (it.get("cluster","").rstrip("/").split("/")[-1]
                if "/clusters/" in str(it.get("cluster","")) else None))

def scrape_third_circuit(limit=None, out_csv="third_circuit_cases.csv", checkpoint_every=200):
    # Resume: load any already-saved cluster ids
    saved_cids = set()
    if os.path.exists(out_csv):
        try:
            for chunk in pd.read_csv(out_csv, usecols=["_cid"], chunksize=10000):
                saved_cids.update(map(str, chunk["_cid"].dropna().astype(str)))
        except Exception:
            pass

    rows_buffer = []
    url, params = BASE_SEARCH, {
        "court":"ca3", "type":"o",
        "page_size":100,
        "fields":"id,cluster_id,cluster,caseName,docketNumber,dateFiled"
    }

    def flush_buffer():
        if not rows_buffer:
            return
        df_chunk = pd.DataFrame(rows_buffer, columns=["_cid","name","docket_number","decision_date","opinion_text"])
        header_needed = not os.path.exists(out_csv)
        _atomic_append_csv(out_csv, df_chunk, header_needed)
        rows_buffer.clear()

    try:
        total_kept = len(saved_cids)
        while url and (limit is None or total_kept < limit):
            data = _get_json(url, params=params); params=None
            for it in data.get("results", []):
                if limit is not None and total_kept >= limit: break

                cid = _robust_cluster_id(it)
                if not cid: continue
                scid = str(cid)
                if scid in saved_cids: continue

                text = _cluster_text(cid)
                if not text: continue

                name = (it.get("caseName") or "").strip()
                decision = it.get("dateFiled") or ""
                docket = it.get("docketNumber") or ""
                if not docket:
                    try:
                        cl_meta = _get_json(CLUSTER_URL.format(id=cid), params={"fields":"docket"})
                        d_url = cl_meta.get("docket")
                        if d_url:
                            d = _get_json(d_url, params={"fields":"docket_number,docket_number_core"})
                            docket = d.get("docket_number") or d.get("docket_number_core") or ""
                    except Exception:
                        pass

                rows_buffer.append({
                    "_cid": scid,
                    "name": name,
                    "docket_number": docket,
                    "decision_date": decision,
                    "opinion_text": text
                })
                saved_cids.add(scid)
                total_kept += 1

                if len(rows_buffer) >= checkpoint_every:
                    flush_buffer()

            url = data.get("next")

        flush_buffer()

        if os.path.exists(out_csv):
            df = pd.read_csv(out_csv)
            if "_cid" in df.columns:
                df = df[["name","docket_number","decision_date","opinion_text"]]
            return df
        else:
            df = pd.DataFrame(rows_buffer)
            if not df.empty:
                df = df[["name","docket_number","decision_date","opinion_text"]]
            return df

    except KeyboardInterrupt:
        flush_buffer()
        df = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["name","docket_number","decision_date","opinion_text"])
        if "_cid" in df.columns:
            df = df[["name","docket_number","decision_date","opinion_text"]]
        return df
    
import re, pandas as pd, unicodedata

# def extract_district_judge_info(cl_data: pd.DataFrame, judges_info: pd.DataFrame) -> pd.DataFrame:
#     """Add 'district judge' (last name, lowercase) and 'district judge id' (Int64) to cl_data."""
#     # --- helpers ---
#     def _norm(s):  # normalize for matching
#         return unicodedata.normalize("NFKD", str(s or "")).strip()

#     def _canon(s):  # canonical key: lowercase letters only
#         return re.sub(r"[^a-z]", "", _norm(s).lower())

#     def _strip_honorifics(s):
#         s = re.sub(r"(?i)\b(the\s+)?honorable\b|^hon\.?\s*", "", _norm(s))
#         s = re.sub(r"\s+", " ", s).strip(" ,;")
#         return s

#     SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
#     def _split_name(full):
#         # keep only word-ish tokens; drop suffixes at end
#         toks = re.findall(r"[A-Za-z][A-Za-z'\.-]*", _strip_honorifics(full))
#         while toks and toks[-1].rstrip(".").lower() in SUFFIX:
#             toks.pop()
#         if not toks: 
#             return "", ""
#         first = re.sub(r"\.$", "", toks[0])
#         last  = re.sub(r"\.$", "", toks[-1])
#         return first, last

#     # patterns (non-greedy, stop at line end; forbid digits in names)
#     # 1) "District Judge: Hon. Jennifer P. Wilson"
#     pat_after = re.compile(
#         r"(?im)\bdistrict\s+judge\b[:\s,]*"
#         r"(?:(?:the\s+)?honorable|hon\.)?\s*"
#         r"(?P<name>[A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*$"
#     )
#     # 2) "SCIRICA, District Judge."
#     pat_before = re.compile(
#         r"(?im)(?P<name>[A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:U\.S\.\s*)?district\s+judge\b"
#     )
#     # 3) looser single-line fallback near "District Judge"
#     pat_line = re.compile(
#         r"(?im)^(?P<line>.*\bdistrict\s+judge\b.*)$"
#     )
#     # court: "United States District Court for the Middle District of Pennsylvania"
#     pat_court = re.compile(
#         r"(?im)(?:on\s+)?appeal\s+from\s+the\s+united\s+states\s+district\s+court\s+for\s+the\s+(?P<court>[^\r\n,]+)"
#     )

#     def _extract_name(txt: str) -> str:
#         if not txt: return ""
#         # prefer exact one-line “after” form
#         for m in pat_after.finditer(txt):
#             return _strip_honorifics(m.group("name"))
#         # then “before” form
#         for m in pat_before.finditer(txt):
#             return _strip_honorifics(m.group("name"))
#         # fallback: scan the first line that contains "district judge" and try to pick a clean name nearby
#         m = pat_line.search(txt)
#         if m:
#             line = m.group("line")
#             # try ": <name>" after the phrase
#             m2 = re.search(
#                 r"(?i)\bdistrict\s+judge\b[:\s,]*"
#                 r"(?:(?:the\s+)?honorable|hon\.)?\s*([A-Z][A-Za-z'\.\- ]*[A-Za-z])", line)
#             if m2: return _strip_honorifics(m2.group(1))
#             # or "<NAME>, District Judge"
#             m3 = re.search(
#                 r"([A-Z][A-Za-z'\.\- ]*[A-Za-z])\s*,\s*(?:U\.S\.\s*)?district\s+judge\b", line, flags=re.I)
#             if m3: return _strip_honorifics(m3.group(1))
#         return ""

#     def _extract_court(txt: str) -> str:
#         if not txt: return ""
#         m = pat_court.search(txt)
#         if m: return _norm(m.group("court"))
#         # softer fallback if phrasing differs slightly
#         m2 = re.search(r"(?im)united\s+states\s+district\s+court\s+for\s+the\s+([^\r\n,]+)", txt)
#         return _norm(m2.group(1)) if m2 else ""

#     # prep judge reference (canonical keys)
#     J = judges_info.copy()
#     J["jid"]       = pd.to_numeric(J["judge id"], errors="coerce").astype("Int64")
#     J["last_key"]  = J["last name"].map(_canon)
#     J["first_key"] = J["first name"].map(lambda x: _canon(re.sub(r"\.$","", (str(x).split() or [""])[0])))
#     J["court_key"] = J["court name"].map(_canon)

#     def _resolve_id(full_name: str, court_str: str) -> pd.Series:
#         first, last = _split_name(full_name)
#         last_key  = _canon(last)
#         first_key = _canon(first)
#         court_key = _canon(court_str)
#         if not last_key:
#             return pd.Series({"district judge": "", "district judge id": pd.NA})

#         cand = J[J["last_key"] == last_key]
#         if len(cand) == 1:
#             return pd.Series({"district judge": last.lower(), "district judge id": cand["jid"].iloc[0]})
#         if len(cand) > 1 and first_key:
#             cand2 = cand[cand["first_key"] == first_key]
#             if len(cand2) == 1:
#                 return pd.Series({"district judge": last.lower(), "district judge id": cand2["jid"].iloc[0]})
#             if len(cand2) > 1 and court_key:
#                 c3 = cand2[cand2["court_key"].apply(lambda x: bool(x) and (court_key in x or x in court_key))]
#                 if len(c3):
#                     return pd.Series({"district judge": last.lower(), "district judge id": c3["jid"].iloc[0]})
#         # if still ambiguous or no match
#         return pd.Series({"district judge": last.lower(), "district judge id": pd.NA})

#     # apply per row (robustness > minimal lines)
#     out = cl_data.copy()
#     texts = out["opinion_text"].astype(str)
#     names = texts.map(_extract_name)
#     courts = texts.map(_extract_court)
#     resolved = [ _resolve_id(n, c) for n, c in zip(names, courts) ]
#     resolved_df = pd.DataFrame(resolved, index=out.index)
#     out["district judge"] = resolved_df["district judge"]
#     out["district judge id"] = resolved_df["district judge id"].astype("Int64")
#     return out


import re, unicodedata
import pandas as pd

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


def cl_loader(judges):
    thrd_judges = judges[judges['court name'].str.contains(r"Third|Delaware|New Jersey|Pennsylvania|Virgin Islands")]
    cl_data = pd.read_csv('third_circuit_cases.csv') if os.path.exists('third_circuit_cases.csv') else pd.DataFrame(columns=["name","docket_number","decision_date","opinion_text"])

    PHRASE = r"(?i)(?<!\w)(?:on[\s\u00A0]+)?appeal[\s\u00A0]+from(?!\w)|(?<!\w)district[\s\u00A0]+judge(?!\w)"
    cl_clean = cl_data[cl_data["opinion_text"].str.contains(PHRASE, case=False, regex=True, na=False)].reset_index(drop=True)
    cl_extracted = extract_district_judge_info(cl_clean, judges)

    cl_clean = cl_extracted[cl_extracted['district judge id'].notna()]
    cl_clean = cl_extracted[cl_extracted["district judge id"].isin(thrd_judges["judge id"].dropna().astype(int))]
    cl_clean['is_appellate'] = 1
    cl_clean['unique_id'] = 'CL_' + cl_clean.index.astype(str)
    cl_clean = cl_clean[cl_clean['district judge id'].notna()]

    return cl_clean