"""
This file runs the batch inference using the OpenAI API.
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib        import Path
from typing         import Optional, Tuple, List
import pandas       as pd

from dotenv         import load_dotenv
from openai         import OpenAI

from data_loading   import build_cap_dataset, text_builder, truncate_opinion

system_msg          = (
    "You are a legal assistant. Your task has seven parts, and you must return JSON only with the required keys.\n\n"
    "1) Disposition (opinion outcome). Choose exactly one of: affirmed, reversed, vacated, remanded, reversed and remanded, modified.\n\n"
    "2) Case type. Choose exactly one of: criminal, civil, tort, labor, contract, bankruptcy, other.\n\n"
    "3) Error category (primary grounds). Choose exactly one of: legal error, procedural error, insufficient evidence, constitutional violation, abuse of discretion, new or controlling precedent.\n\n"
    "4) Classify the petitioner and respondent each into exactly one of: male, female, group of individuals, company, other.\n\n"
    "5) Politicality of the case: choose an integer from 1 to 5. This number tells us how politically important or controversial the case is at the appellate court level.\n"
    "   - 1 = Very Low. Routine or technical matters with no political or public interest.\n"
    "     Example: a bankruptcy dispute or a contract interpretation between two small companies.\n\n"
    "   - 2 = Low. Cases involving government rules or workplace disputes, but not really political or ideological.\n"
    "     Example: a worker suing for unpaid overtime, or a minor administrative regulation challenge.\n\n"
    "   - 3 = Moderate. Cases with visible policy or social issues that might get regional media coverage or advocacy group attention.\n"
    "     Examples: environmental regulations, civil rights in employment, disability rights in schools.\n\n"
    "   - 4 = High. Cases raising significant constitutional or statutory questions tied to politically sensitive issues.\n"
    "     Examples: challenges to state abortion restrictions, gun laws, voting access, or immigration enforcement.\n\n"
    "   - 5 = Very High. Cases that, while still in the appellate courts, touch on major nationwide controversies or issues likely to be considered by the Supreme Court later.\n"
    "     Examples: disputes over redistricting and voting rights, statewide abortion bans, or immigration policy at the federal level.\n\n"
    "6) Profile level of the case: choose low, medium, or high. This measures how prominent the people or organizations involved are.\n"
    "   - Low = ordinary people, small local disputes, no broader significance.\n"
    "     Example: two neighbors suing each other over a fence.\n\n"
    "   - Medium = recognizable companies, regional groups, or moderately well-known individuals.\n"
    "     Example: a lawsuit involving a mid-sized company or a regional university.\n\n"
    "   - High = very large corporations, national organizations, famous public figures, or cases that received widespread publicity.\n"
    "     Example: Apple, the ACLU, a well-known politician, or a celebrity.\n\n"
    "7) Lower-court judge name: return this only if the text of the appellate opinion clearly names the judge from the court BELOW (the district or trial court) whose decision is being reviewed.\n"
    "   - Extract the judge from the court BELOW this appeal (e.g., federal district judge in a federal appeal; state trial judge in a state appeal).\n"
    "   - ONLY if the opinion text explicitly names that lower-court judge, return their first and last name.\n"
    "   - If not explicitly named, set BOTH fields to null. Do not guess, infer, or use outside knowledge.\n"
    "   - Ignore appellate panel judges (e.g., Circuit Judges), concurrences/dissents, and counsel.\n"
    "   - Strip honorifics/titles (e.g., 'Hon.', 'Judge', 'J.', 'C.J.', 'Sr.'). Keep just first and last token.\n"
    "   Examples: 'the district court (Judge Jane A. Smith) held' -> first='Jane', last='Smith'; 'Hon. Robert P. Jones' -> first='Robert', last='Jones'.\n\n"
    "Return only valid JSON. Do NOT add explanations or extra keys. If you cannot determine any field with high confidence, set that field to null (do NOT guess). \n"
    "The output JSON must contain exactly nine keys: 'opinion', 'case_type', 'error_category', 'petitioner_type', 'respondent_type', 'politicality_score', 'profile_level', 'lower_judge_first', 'lower_judge_last'."
)

developer_msg       = (
    "Always output valid JSON only. No explanations, no extra keys. Do not guess names. Exactly nine keys: 'opinion', 'case_type', 'error_category', 'petitioner_type', 'respondent_type', 'politicality_score', 'profile_level', 'lower_judge_first', 'lower_judge_last'."
)

user_template       = "Classify this appellate opinion:\n\n{opinion_text}"

class batch_request_body:
    def __init__(self, opinion_text: str):
        self.model          = "gpt-5-mini-2025-08-07"
        self.messages       = [
            {"role": "system", "content":       system_msg},
            {"role": "developer", "content":    developer_msg},
            {"role": "user", "content":         user_template.format(opinion_text=opinion_text)},
        ]

    def to_dict(self):
        return {
            "model": self.model,
            "messages": self.messages,
        }

class BatchRequest:
    def __init__(self, 
                 custom_id: str, # case ID
                 body: str):
        self.custom_id      = str(custom_id)                
        self.method         = "POST"                           
        self.url            = "/v1/chat/completions"               
        self.body           = batch_request_body(body).to_dict()  

    def to_dict(self):
        return {
            "custom_id":    self.custom_id,
            "method":       self.method,
            "url":          self.url,
            "body":         self.body,
        }

def _extract_text(resp: dict):
    if not isinstance(resp, dict):
        return None
    body = resp.get("body")
    if isinstance(body, dict):
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            return choices[0]["message"]["content"]
    # responses API shapes (fallbacks)
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    out = resp.get("output")
    if isinstance(out, list):
        parts = []
        for msg in out:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    t = part.get("text")
                    if t: parts.append(t)
        return "".join(parts) if parts else None
    return None

def _collect_done_ids(output_file: Path) -> set[str]:
    """Returns set of case ids that have already been processed in the final output file."""
    done_ids = set()
    if not output_file.exists():
        return done_ids
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj     = json.loads(line)
                cid     = obj.get("custom_id")
                if cid:
                    done_ids.add(cid)
            except Exception:
                continue
    return done_ids

def _build_full_input(df, 
                      inp_path: Path,
                      mx_tk:    int = 3000) -> int: 
    """Writes the full input jsonl file for the API from the cases dataframe.
    
    :param df:         DataFrame containing the cases to process.
    :param out_path:   Path to write the output jsonl file. # input_api_format.jsonl
    :param mx_tk:      Maximum tokens for each case text.
    """

    cases  = text_builder(df, limit=None, mx_tk=mx_tk) # should be +- 50.000 for Third Circuit
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with inp_path.open("w", encoding="utf-8") as f:
        for row in cases:
            cid     = str(row["id"])
            body    = row["text"]
            br      = BatchRequest(custom_id=cid, body=body)
            json.dump(br.to_dict(), f, ensure_ascii=False)
            f.write("\n")
            n += 1
    print(f"[Build] Wrote {n} requests -> {inp_path}")
    return n    

def _write_missing_only(df, merged_done: set[str], out_path: Path, mx_tk: int = 3000) -> int:
    """Write ONLY missing requests to JSONL by reusing text_builder output."""
    cases           = text_builder(df, limit=None, mx_tk=mx_tk)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    id2text         = {str(r["id"]): r["text"] for r in cases if "id" in r and "text" in r}
    missing_ids     = set(id2text.keys()) - merged_done
    cnt             = 0

    with out_path.open("w", encoding="utf-8") as f:
        for cid in sorted(missing_ids):
            br = BatchRequest(custom_id=cid, body=id2text[cid])
            json.dump(br.to_dict(), f, ensure_ascii=False)
            f.write("\n")
            cnt += 1

    print(f"[Build] Missing requests: {cnt} -> {out_path}")
    return cnt

def _split_by_size(src:         Path, 
                   dst_dir:     Path, 
                   prefix:      str = "input_chunk", 
                   max_bytes=   200*1024*1024) -> List[Path]:
    """Now that we have the correct input for the API (input_file), we need to split it up in parts no larger than 200MB.
    This returns parth_paths, in the input_dir, which we will upload to OpenAI (batch_runs/input).
    """

    dst_dir.mkdir(parents=True, exist_ok=True)
    parts:                      List[Path] = []
    part_idx, bytes_in_part     = 1, 0
    part_path                   = dst_dir / f"{prefix}_{part_idx}.jsonl"
    out_f                       = part_path.open("wb")
    
    with src.open("rb") as inp:
        for line in inp:
            ln = len(line)
            if bytes_in_part > 0 and bytes_in_part + ln > max_bytes:
                out_f.close()
                parts.append(part_path)
                part_idx += 1
                bytes_in_part = 0
                part_path = dst_dir / f"{prefix}_{part_idx}.jsonl"
                out_f = part_path.open("wb")
            out_f.write(line)
            bytes_in_part += ln
    out_f.close()
    parts.append(part_path)
    print(f"[Split] Created {len(parts)} part(s) in {dst_dir}")
    return parts

def _enqueue_chunks(parts: List[Path], endpoint: str, completion_window: str) -> List[str]:
    """Upload chunk files and create batches."""
    load_dotenv()
    client      = OpenAI()
    batch_ids:  List[str] = []
    for p in parts:
        with p.open("rb") as fh:
            file_obj = client.files.create(file=fh, purpose="batch")
        batch = client.batches.create(
            input_file_id       = file_obj.id,
            endpoint            = endpoint,
            completion_window   = completion_window,
            metadata={"description": f"Batch for {p.name}"},
        )
        batch_ids.append(batch.id)
        print(f"[Batch] Created {batch.id} for {p.name}")
    return batch_ids

def _download_new_outputs(batch_ids: List[str], out_dir: Path, poll: bool, poll_interval: int = 20) -> List[Path]:
    """Optionally poll batches to completion and download outputs/errors."""
    load_dotenv()
    client = OpenAI()
    out_dir.mkdir(parents=True, exist_ok=True)
    terminal = {"completed", "failed", "cancelled", "expired"}
    new_outputs: List[Path] = []

    for bid in batch_ids:
        if poll:
            while True:
                b = client.batches.retrieve(bid)
                st = getattr(b, "status", None)
                print(f"[Poll] {bid} status={st}")
                if st in terminal:
                    break
                time.sleep(poll_interval)
        else:
            b = client.batches.retrieve(bid)

        out_id = getattr(b, "output_file_id", None)
        if out_id:
            out_path = out_dir / f"{bid}_results.jsonl"
            if not out_path.exists():
                stream = client.files.content(out_id)
                with out_path.open("wb") as f:
                    f.write(stream.read())
                print(f"[Download] {bid} -> {out_path}")
                new_outputs.append(out_path)

        err_id = getattr(b, "error_file_id", None)
        if err_id:
            err_path = out_dir / f"{bid}_errors.jsonl"
            if not err_path.exists():
                estream = client.files.content(err_id)
                with err_path.open("wb") as f:
                    f.write(estream.read())
                print(f"[Errors]   {bid} -> {err_path}")

    return new_outputs

def _append_and_dedupe(merged_path: Path, newly_downloaded: List[Path]) -> Tuple[int, int]:
    """
    Append new results to merged_path, then rewrite merged_path with unique custom_ids.
    Returns (kept, dropped_dupes).
    """
    # Append
    if newly_downloaded:
        with merged_path.open("ab" if merged_path.exists() else "wb") as out_f:
            for p in newly_downloaded:
                with p.open("rb") as inp:
                    out_f.write(inp.read())
        print(f"[Append] Added {len(newly_downloaded)} file(s) to {merged_path}")

    # Deduplicate by custom_id (keep first occurrence)
    seen: set[str] = set()
    kept_lines: List[str] = []
    dropped = 0

    if merged_path.exists():
        with merged_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cid = str(rec.get("custom_id"))
                if cid in seen:
                    dropped += 1
                    continue
                seen.add(cid)
                kept_lines.append(line)

        # Rewrite file with uniques
        tmp = merged_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as out:
            out.writelines(kept_lines)
        tmp.replace(merged_path)

    print(f"[Dedupe] Unique={len(seen)} | Dropped dupes={dropped}")
    return len(seen), dropped

def load_case_results(path: str = "batch_runs/api_responses.jsonl") -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            if rec.get("error"):
                continue
            try:
                content = rec["response"]["body"]["choices"][0]["message"]["content"]
                obj = json.loads(content)  # parse the 9-key JSON
            except Exception:
                continue

            # Attach your own ID (custom_id)
            obj["custom_id"] = rec.get("custom_id")

            records.append(obj)

    return pd.DataFrame.from_records(records)

# --- main one-call function -------------------------------------------------
def run_incremental_batches(
    df,
    *,
    work_dir:               str | Path = "batch_runs",
    full_input_name:        str = "input_api_format.jsonl",
    missing_input_name:     str = "input_missing.jsonl",
    endpoint:               str = "/v1/chat/completions",
    completion_window:      str = "24h",
    max_bytes:              int = 200 * 1024 * 1024,
    poll:                   bool = False,           # set True if you want to wait & download now
) -> None:
    """
    Full incremental pipeline:
      - build full input JSONL (for reference/inspection)
      - read existing merged outputs: batch_runs/api_responses.jsonl
      - create input_missing.jsonl (only not-yet-answered custom_ids)
      - split, upload, create batches
      - optionally poll + download
      - append new outputs into api_responses.jsonl, then dedupe
    """
    work = Path(work_dir)
    inputs_dir  = work / "input_chunks"
    outputs_dir = work / "output"
    merged_path = work / "api_responses.jsonl"
    full_input  = work / full_input_name
    miss_input  = work / missing_input_name

    # 1: Build full input JSONL
    ###############################################################################################
    total               = _build_full_input(df, full_input) 

    # 2: Already-done custom_ids
    ###############################################################################################
    done_ids            = _collect_done_ids(merged_path)
    print(f"[State] Already answered: {len(done_ids)} / {total}")

    # 3: Build missing-only JSONL
    ###############################################################################################
    new_count           = _write_missing_only(df, done_ids, miss_input)
    if new_count == 0:
        print("[Done] Everything already answered. Just dedup/clean the merged file.")
        _append_and_dedupe(merged_path, newly_downloaded=[])
        return

    # 4: Split missing-only file into <=200MB chunks
    ###############################################################################################
    parts               = _split_by_size(miss_input, inputs_dir, prefix="input_chunk", max_bytes=max_bytes)

    # 5: Enqueue batches
    ###############################################################################################
    batch_ids           = _enqueue_chunks(parts, endpoint=endpoint, completion_window=completion_window)
    print(f"[Enqueued] {len(batch_ids)} batch(es): {batch_ids}")

    # 6: Optionally poll/download now, then append & dedupe
    ###############################################################################################
    new_outputs         = _download_new_outputs(batch_ids, outputs_dir, poll=poll)
    kept, dropped       = _append_and_dedupe(merged_path, newly_downloaded=new_outputs)
    print(f"[Final] merged={merged_path} | unique={kept} | dropped_dupes={dropped}")
# ============================================================================

# ---------- Config ----------
INPUT_PATH          = Path("batch_runs/overlap_input.jsonl")   # already created
OUTPUT_DIR          = Path("batch_runs/overlap_outputs")       # NEW folder for this run
MERGED_RESULTS_PATH = OUTPUT_DIR / "overlap_results.jsonl"     # NEW results file
ENDPOINT            = "/v1/chat/completions"                    # or "/v1/responses" if you used Responses API in the input
COMPLETION_WINDOW   = "24h"                                     # typical batch window; adjust as needed
POLL                = True                                      # set False if you don't want to poll
POLL_INTERVAL       = 20                                        # seconds


def enqueue_single_file(inp: Path, endpoint: str, completion_window: str) -> str:
    """Upload one JSONL and create a single batch. Returns batch_id."""
    load_dotenv()
    client = OpenAI()
    with inp.open("rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id     = file_obj.id,
        endpoint          = endpoint,
        completion_window = completion_window,
        metadata          = {"description": f"Batch for {inp.name}"},
    )
    print(f"[Batch] Created {batch.id} for {inp.name}")
    return batch.id


def download_outputs_for_batch(batch_id: str, out_dir: Path, poll: bool, poll_interval: int = 20) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Optionally poll a single batch to completion, then download outputs/errors.
    Returns (results_path, errors_path) — either may be None if not present.
    """
    load_dotenv()
    client = OpenAI()
    out_dir.mkdir(parents=True, exist_ok=True)

    terminal = {"completed", "failed", "cancelled", "expired"}
    while True:
        b = client.batches.retrieve(batch_id)
        st = getattr(b, "status", None)
        print(f"[Poll] {batch_id} status={st}")
        if not poll or st in terminal:
            break
        time.sleep(poll_interval)

    results_path = None
    errors_path  = None

    out_id = getattr(b, "output_file_id", None)
    if out_id:
        results_path = out_dir / f"{batch_id}_results.jsonl"
        if not results_path.exists():  # don't clobber if re-running
            stream = client.files.content(out_id)
            with results_path.open("wb") as f:
                f.write(stream.read())
        print(f"[Download] results -> {results_path}")

    err_id = getattr(b, "error_file_id", None)
    if err_id:
        errors_path = out_dir / f"{batch_id}_errors.jsonl"
        if not errors_path.exists():
            stream = client.files.content(err_id)
            with errors_path.open("wb") as f:
                f.write(stream.read())
        print(f"[Download] errors  -> {errors_path}")

    return results_path, errors_path


def merge_fresh_results(merged_path: Path, batch_result_files: List[Path]) -> Tuple[int, int]:
    """
    Write a brand-new merged file only from the provided batch_result_files.
    Dedupes by custom_id within these files. Does NOT read or touch old outputs.
    Returns (unique_kept, dropped_dupes_within_these_files).
    """
    seen: set[str] = set()
    kept_lines: List[str] = []
    dropped = 0

    for p in batch_result_files:
        if not p or not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cid = str(rec.get("custom_id"))
                if cid in seen:
                    dropped += 1
                    continue
                seen.add(cid)
                kept_lines.append(line)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as out:
        out.writelines(kept_lines)

    print(f"[Merge] Wrote {merged_path} | Unique={len(seen)} | Dropped dupes={dropped}")
    return len(seen), dropped



if __name__ == "__main__":
    df = build_cap_dataset()
    run_incremental_batches(
        df,
        work_dir="batch_runs",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        max_bytes=200*1024*1024,
        poll=False,   # set True if you want to wait for completion & auto-download now
    )

    # overlap api call
    # input_path = Path("batch_runs/overlap_input.jsonl")
    # cl = pd.read_csv('third_circuit_on_appeal.csv')
    # cl = cl[cl['docket_number'].notna()] # remove nas
    # cl = cl.drop_duplicates(subset="docket_number", keep="first") # drop dupes
    # cl_non_overlap = cl[cl['overlap_by_substring']==False]
    # cl_non_overlap = cl_non_overlap.rename(columns={'combined_preview': 'opinion_text', 'cluster_id': 'unique_id'})
    # cl_non_overlap['is_appellate'] = 1
    # _build_full_input(cl_non_overlap, input_path)
    # batch_id = enqueue_single_file(INPUT_PATH, ENDPOINT, COMPLETION_WINDOW)

    # # 2) (optional) poll & download this batch's outputs/errors ONLY into the new folder
    # res_path, err_path = download_outputs_for_batch(batch_id, OUTPUT_DIR, poll=POLL, poll_interval=POLL_INTERVAL)

    # # 3) Merge just-downloaded results into a new, isolated merged file
    # #    (This does NOT read any old outputs; it only uses this run's files.)
    # present = [p for p in [res_path] if p is not None]
    # merge_fresh_results(MERGED_RESULTS_PATH, present)

    # print(f"\nDone. Your fresh results are in: {MERGED_RESULTS_PATH}")