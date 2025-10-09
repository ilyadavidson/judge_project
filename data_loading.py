import os, re, glob
import numpy as np
import pandas as pd

import pyarrow.dataset as ds
import pyarrow.compute as pc
import tiktoken
import re
import unicodedata
import pandas as pd
from typing import Optional
import json
from helper_functions import norm_id
from typing import Dict, List, Optional
import os, io, re, time, random, requests, pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tempfile import NamedTemporaryFile  

def _arrow_filter_table(files: list[str], courts: list[str]): # saved in scr/jp/cap/
    """
    Use PyArrow Dataset to scan many Parquet files while filtering over smaller subset.

    :param files: parquet files.
    :param courts: courts included in the final df.
    """
    dataset         = ds.dataset(files, format="parquet")

    courts_regex    = rf"(?i)({'|'.join(map(re.escape, courts))})"
    filt            = pc.match_substring_regex(ds.field("court_name"), courts_regex)

    table           = dataset.to_table(filter=filt, use_threads=True)
    return table

def build_cap_dataset(
        pattern     = "data/parquet_files/CAP_data_*.parquet",
        appellate   = ["Third Circuit"], 
        district    = ["Delaware", "New Jersey", "Pennsylvania", "Virgin Islands"]):
    """
    Builds the dataframe off of the parquet files. Due to storage there's an option to only load in certain circuits.
    Default is set to the third circuit.
    Put either appellate or district as None to get the whole df. 
    """
    
    # Loads in the parquet files
    ############################################################################################################
    files = sorted(glob.glob(pattern))
    print(f"Working dir: {os.getcwd()}")
    print(f"Found {len(files)} parquet files for pattern: {pattern}")

    # Make a sub-set if needed
    ############################################################################################################
    if appellate is None or district is None:
        courts      = None
        dataset     = ds.dataset(files, format="parquet")
        table       = dataset.to_table(use_threads=True)
    else:
        courts      = appellate + district
        table       = _arrow_filter_table(files, courts=courts)

    if table.num_rows == 0:
        return pd.DataFrame()

    df = table.to_pandas(use_threads=True)

    # Creates an unique id and appellate identifier
    ############################################################################################################
    df["unique_id"]         = df.index.astype(str)
    df["is_appellate"]      = np.where(df["court_name"].str.contains("Appeals", case=False, na=False), 1, 0)

    # Delete duplicates based on opinion type and docket number and keeps the one with majority vote
    ############################################################################################################
    appellate = (
        df[df["is_appellate"] == 1]
        .sort_values("opinion_type", key=lambda s: s.str.lower().eq("majority"), ascending=False)
        .drop_duplicates(subset=["docket_number"], keep="first")
    )

    non_appellate = df[df["is_appellate"] != 1]

    df            = pd.concat([appellate, non_appellate], ignore_index=True)

    return df

enc = tiktoken.get_encoding("o200k_base")

def truncate_opinion(text, max_tokens= 6000) -> str:
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

def court_listener_cleaner(
    df: pd.DataFrame,                     # court_listener dataframe
    judges_info: Optional[pd.DataFrame] = None,
    text_col: str = "combined_preview",
    source_df: Optional[pd.DataFrame] = None,   # your "cases" df
    source_docket_col: str = "docket_number",
    source_uid_col: str = "unique_id",
) -> pd.DataFrame:
    """
    Parse CourtListener preview text to extract district judge & court, then map to judge id.

    Adds these columns to df:
      - 'district_judge'         : cleaned full name (suffix kept)
      - 'district_judge_clean'   : last name only, lowercase (robust)
      - 'court_name'             : 'for the …' line after 'On Appeal from …'
      - 'judge id'               : nullable Int64 from judges_info (if provided)

    Parameters
    ----------
    df : DataFrame with at least the preview text column (default 'combined_preview').
    judges_info : DataFrame with columns ['judge id','first name','last name','court name'] (optional).
    text_col : name of the text column to parse.

    Returns
    -------
    DataFrame with new columns (original columns preserved).
    """

    # ---------- patterns ----------
    _PAT_JUDGE_LINE = re.compile(r'(?is)District\s+Judge:\s*([^\r\n]+)')
    _PAT_COURT_LINE = re.compile(r'(?is)On\s+Appeal\s+from.*?\n\s*(for\s+the[^\n(]+)')

    _TITLES     = re.compile(r'(?i)^\s*(the\s+honorable|hon\.?|honorable|chief)\s+')
    _MARKERS    = re.compile(r'[\*\u2020\u2021]')  # * † ‡ anywhere
    _PAREN_JUNK = re.compile(r'(?is)\(\s*(?:ret\.?|retired|senior(?:\s+judge)?|emeritus|by\s+designation|pro\s*tem|visiting|acting)[^)]*\)')
    _SUFFIXES   = re.compile(r'(?i)[,\s]+(jr\.?|sr\.?|junior|senior|ii|iii|iv|v)\s*$')

    # --- court canonicalization helpers ---
    _STOP_PHRASES = [
        r'\bthe\b', r'\bfor\b', r'\bof\b', r'\bthe\b',
        r'united\s+states\s+district\s+court', r'u\.s\.\s+district\s+court',
        r'united\s+states\s+court\s+of\s+appeals', r'court\s+of\s+appeals',
        r'district\s+court', r'circuit\s+court'
    ]
    _STOP_RE = re.compile('|'.join(_STOP_PHRASES), flags=re.I)

    # ---------- tiny helpers ----------
    def _fold(s: str) -> str:
        s = unicodedata.normalize('NFKD', s or '')
        return ''.join(ch for ch in s if not unicodedata.combining(ch))

    def _clean_full_name_keep_suffix(s: str) -> str:
        s = (s or '').strip()
        if not s:
            return ''
        s = _TITLES.sub('', s)
        s = _MARKERS.sub('', s)
        s = _PAREN_JUNK.sub('', s)
        s = re.sub(r'\s+', ' ', s).strip(' ,;')
        return s

    def _is_initial(tok: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]\.", tok))

    def _clean_tokens(name: str) -> list[str]:
        name = name.replace(',', ' ')
        toks = name.split()
        out = []
        for t in toks:
            t = _fold(t).strip()
            if not t:
                continue
            keep = re.sub(r"[^A-Za-z'\-]", '', t).strip("-'")
            if keep:
                out.append(keep)
        return out

    def _last_name_only(full: str) -> str:
        if not isinstance(full, str) or not full.strip():
            return ''
        s = _clean_full_name_keep_suffix(full)
        s = _SUFFIXES.sub('', s).strip()
        toks = _clean_tokens(s)
        for t in reversed(toks):
            if not _is_initial(t) and len(t) >= 2:
                return t.lower()
        return ''

    def _first_name_guess(full: str) -> str:
        if not isinstance(full, str) or not full.strip():
            return ''
        s = _clean_full_name_keep_suffix(full)
        s = _SUFFIXES.sub('', s).strip()
        toks = _clean_tokens(s)
        for t in toks:
            if not _is_initial(t) and len(t) >= 2:
                return t.lower()
        return ''

    def _letters_only_key(s: str) -> str:
        return re.sub(r'[^a-z]', '', _fold(s).lower())

    def _first_token_key(s: str) -> str:
        s = (s or '').strip()
        if not s:
            return ''
        tok = next((t for t in re.split(r'[\s,]+', s) if t), '')
        tok = re.sub(r'\.$', '', tok)
        return _letters_only_key(tok)

    def _canon_court(s: str) -> str:
        s = _fold(s).lower()
        s = _STOP_RE.sub(' ', s)
        s = re.sub(r'[\(\)\.,;:]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # ---------- extraction ----------
    def _extract_court_name(text: str) -> str:
        m = _PAT_COURT_LINE.search(text or "")
        if not m:
            return ""
        court = m.group(1).strip()
        return court if court.endswith('.') else court + '.'

    def _extract_district_judge_full(text: str) -> str:
        m = _PAT_JUDGE_LINE.search(text or "")
        if not m:
            return ""
        return _clean_full_name_keep_suffix(m.group(1))

    # ---------- judge id resolution ----------
    def _ensure_helper_cols(ji: pd.DataFrame) -> pd.DataFrame:
        ji = ji.copy()
        # Expect: 'judge id','first name','last name','court name'
        for col in ("last name", "first name", "court name"):
            if col not in ji.columns:
                ji[col] = ''
            else:
                ji[col] = ji[col].fillna('')
        ji['last_name_key']  = ji['last name'].map(_letters_only_key)
        ji['first_name_key'] = ji['first name'].map(_first_token_key)
        ji['court_key']      = ji['court name'].map(_canon_court)
        ji['judge id']       = pd.to_numeric(ji.get('judge id', pd.Series(dtype='float')), errors='coerce').astype('Int64')
        return ji

    def _resolve_judge_id(dj_full: str, court_name: str, judges_info: pd.DataFrame):
        if judges_info is None or judges_info.empty:
            return pd.NA
        ji = _ensure_helper_cols(judges_info)
        last_key  = _letters_only_key(_last_name_only(dj_full))
        first_key = _first_token_key(_first_name_guess(dj_full))
        court_key = _canon_court(court_name or '')

        cand = ji[ji['last_name_key'] == last_key]
        if len(cand) == 0:
            return pd.NA
        # try first name
        cand2 = cand
        if first_key:
            cand_first = cand[cand['first_name_key'] == first_key]
            cand2 = cand_first if len(cand_first) > 0 else cand
            if len(cand_first) == 1:
                return cand_first['judge id'].iloc[0]
        # try court contains either way
        if court_key:
            mask = cand2['court_key'].str.contains(court_key, na=False) | cand2['court_key'].apply(lambda s: court_key in s)
            cand3 = cand2[mask]
            if len(cand3) == 1:
                return cand3['judge id'].iloc[0]
            if len(cand3) > 1:
                return cand3['judge id'].iloc[0]
        if len(cand2) == 1:
            return cand2['judge id'].iloc[0]
        if len(cand) == 1:
            return cand['judge id'].iloc[0]
        return pd.NA

    # ---------- main apply ----------
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' column not found in df")

    out = df.copy()

    # Extract per row
    def _per_row(text: str):
        dj_full = _extract_district_judge_full(text)
        court   = _extract_court_name(text)
        dj_clean = _last_name_only(dj_full)
        jid = _resolve_judge_id(dj_full, court, judges_info) if judges_info is not None else pd.NA
        return pd.Series({
            "district_judge": dj_full,
            "district_judge_clean": dj_clean,
            "court_name": court,
            "judge id": jid
        })

    extracted = out[text_col].apply(_per_row)
    out = pd.concat([out, extracted], axis=1)

    # Normalize dtype of 'judge id' to nullable Int64
    out["judge id"] = pd.to_numeric(out["judge id"], errors="coerce").astype("Int64")

    if source_df is not None and source_docket_col in source_df.columns:
        # Preload all source dockets + ids
        src_pairs = source_df[[source_docket_col, source_uid_col]].dropna().astype(str).values.tolist()

        def find_unique_id(docket: str):
            if not isinstance(docket, str):
                return pd.NA
            for sd, uid in src_pairs:
                if docket in sd or sd in docket:
                    return uid   # take the first overlapping match
            return pd.NA

        out["unique_id"] = out["docket_number"].astype(str).apply(find_unique_id).astype("string")

    return out

def _load_mapping(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        d = {}
    # normalize keys/vals to strings
    out = {}
    for k, v in d.items():
        nk, nv = norm_id(k), norm_id(v)
        if nk and nv:
            out[nk] = nv
    return out

def promotion_info_judges(judge_info):
    """
    Returns a df of all judges who got promoted from district to appellate courts. Their promotion date is the earliest nomination date that they got to the appellate court. 
    """
    ji                  = judge_info.copy()

    ji["court type"]    = ji["court type"].astype(str).str.lower()

    dj                  = ji[ji["court type"].str.contains("district", na=False)]
    aj                  = ji[ji["court type"].str.contains("appeal|circuit", na=False)]

    aj["nomination date"] = pd.to_datetime(aj["nomination date"], errors="coerce")
    promo_dates = (
        aj.sort_values(["judge id", "nomination date"])
          .groupby("judge id")["nomination date"]
          .first()
    )

    ji_district = (
        dj.drop_duplicates("judge id")
          .assign(
              is_promoted=lambda d: d["judge id"].isin(promo_dates.index).astype(int),
              promotion_date=lambda d: d["judge id"].map(promo_dates)
          )
    )

    return ji_district


import os, io, re, time, random, requests, pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tempfile import NamedTemporaryFile  

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