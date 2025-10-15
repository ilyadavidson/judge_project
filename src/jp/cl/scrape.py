"""
This file scrapes Court Listener for data 
"""

import os, io, re, time, random, requests, pandas   as pd
from dotenv                                         import load_dotenv
from bs4                                            import BeautifulSoup
from pypdf                                          import PdfReader
from tempfile                                       import NamedTemporaryFile  
from jp.utils.text                                  import _norm_circuit, ensure_dir

# SCRAPE TOKENS
###############################################################################
load_dotenv()
BASE_SEARCH = "https://www.courtlistener.com/api/rest/v4/search/"
CLUSTER_URL = "https://www.courtlistener.com/api/rest/v4/clusters/{id}/"
session = requests.Session()
session.headers.update({
    "Authorization": f"Token {os.getenv('COURTLISTENER_TOKEN')}",
    "User-Agent": os.getenv("COURTLISTENER_USER_AGENT","courtlistener-scraper/1.0")
})
RETRY = {429,500,502,503,504}

SCRAPED_DIR = ensure_dir(("data/artifacts/cl/scraped"))

###############################################################################
def _get_json(url, params=None, timeout=60, attempts=10):
    """Get JSON from CourtListener with retries on transient errors."""
    delay       = 0.8
    backoff     = 1.7
    last_err    = None
    for a in range(attempts):
        try:
            r = session.get(url, params=params if a == 0 else None, timeout=timeout)
        except (requests.ReadTimeout, requests.ConnectTimeout, requests.ConnectionError) as e:
            last_err = e
            time.sleep(min(delay * (backoff ** a) + random.uniform(0, 0.6), 30))
            continue

        if r.status_code in RETRY:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    time.sleep(float(ra))
                except Exception:
                    pass
            time.sleep(min(delay * (backoff ** a) + random.uniform(0, 0.6), 30))
            continue

        r.raise_for_status()
        return r.json()

    if last_err:
        raise last_err
    r.raise_for_status()  # will raise the last HTTP error if we got here

def _html_to_text(h):
    """Convert HTML to plain text, removing scripts/styles and excessive newlines."""
    soup=BeautifulSoup(h or "","html.parser")
    for b in soup(["script","style"]): b.decompose()
    return re.sub(r"\n{3,}","\n\n", soup.get_text("\n")).strip()

def _pdf_to_text(b):
    """Extract text from PDF bytes, joining pages."""
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

def scrape_third_circuit(cid, limit=None, checkpoint_every=200):
    court_code = _norm_circuit(cid)   
    out_csv = SCRAPED_DIR / f"{cid}_scraped.csv" 

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
        "court": court_code, 
        "type":"o",
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
            try:
                data = _get_json(url, params=params); params=None
            except Exception as e:
                time.sleep(5)
                continue
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

    return 