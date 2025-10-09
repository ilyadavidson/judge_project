import json
import tempfile
from pathlib import Path
import pandas as pd

"""
This file contains helpers for different data formats.
"""

# ---------- General ----------
def ensure_dir(path: str | Path):
    """Create parent directories if they don't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ---------- CSV / Parquet ----------
def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file safely."""
    return pd.read_parquet(path)

def write_parquet_atomic(df: pd.DataFrame, path: str | Path):
    """Atomically write a Parquet file (avoids partial writes)."""
    ensure_dir(path)
    tmp = Path(tempfile.mktemp(suffix=".parquet", dir=Path(path).parent))
    df.to_parquet(tmp, index=False)
    Path(tmp).replace(path)

def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def write_csv_atomic(df: pd.DataFrame, path: str | Path, **kwargs):
    ensure_dir(path)
    tmp = Path(tempfile.mktemp(suffix=".csv", dir=Path(path).parent))
    df.to_csv(tmp, index=False, **kwargs)
    Path(tmp).replace(path)

# ---------- JSONL ----------
def read_jsonl(path: str | Path) -> list[dict]:
    """Load JSON Lines into a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl_atomic(records: list[dict], path: str | Path):
    """Append JSONL records atomically."""
    ensure_dir(path)
    tmp = Path(tempfile.mktemp(suffix=".jsonl", dir=Path(path).parent))
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Append atomically
    with open(path, "a", encoding="utf-8") as dst, open(tmp, "r", encoding="utf-8") as src:
        for line in src:
            dst.write(line)