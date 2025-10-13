"""
This file runs the batch inference using the OpenAI API.
"""

from __future__ import annotations

import json
import os

from pathlib        import Path


from dotenv         import load_dotenv
from openai         import OpenAI
from typing         import List
import time

from jp.utils.text   import text_builder
load_dotenv()

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
    """
    Constructs the body for a batch request to the OpenAI API.
    """
    
    def __init__(self, case_name: str):
        self.model          = "gpt-4o"
        self.messages       = [
            {"role": "system", "content":       system_msg},
            {"role": "developer", "content":    developer_msg},
            {"role": "user", "content":         user_template.format(book_name=case_name)},
        ]

    def to_dict(self):
        return {
            "model": self.model,
            "messages": self.messages,
        }

class BatchRequest:
    def __init__(self, case_name: str):
        self.custom_id = str(case_name)
        self.method = "POST"
        self.url = "/v1/chat/completions"
        self.body = batch_request_body(case_name).to_dict()

    def to_dict(self):
        return {
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body,
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

def build_input(df, 
                inp_path: str = 'data/artifacts/api/requests/api_requests.jsonl',
                mx_tk:    int = 3000): 
    """Writes the full input jsonl file for the API from the cases dataframe. Only takes the mx_tk from the opinion text to limit input and computing costs.
    
    :param df:         DataFrame containing the cases to process.
    :param out_path:   Path to write the output jsonl file. 
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

def split_by_size(src:         Path, 
                  output_path: Path, 
                  prefix:      str = "input_chunk", 
                  max_bytes=   200*1024*1024) -> List[Path]:
    """Now that we have the correct input for the API (input_file), we need to split it up in parts no larger than 200MB.
    This returns , in the input_dir, which we will upload to OpenAI (batch_runs/input).
    """

    output_path.mkdir(parents=True, exist_ok=True)
    parts:                      List[Path] = []
    part_idx, bytes_in_part     = 1, 0
    part_path                   = output_path / f"{prefix}_{part_idx}.jsonl"
    out_f                       = part_path.open("wb")
    
    with src.open("rb") as inp:
        for line in inp:
            ln = len(line)
            if bytes_in_part > 0 and bytes_in_part + ln > max_bytes:
                out_f.close()
                parts.append(part_path)
                part_idx += 1
                bytes_in_part = 0
                part_path = output_path / f"{prefix}_{part_idx}.jsonl"
                out_f = part_path.open("wb")
            out_f.write(line)
            bytes_in_part += ln
    out_f.close()
    parts.append(part_path)
    print(f"[Split] Created {len(parts)} part(s) in {output_path}")
    return parts

def enqueue_chunks(parts: List[Path], endpoint: str, completion_window: str) -> List[str]:
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

def download_new_outputs(batch_ids: List[str], output_path: Path, poll: bool, poll_interval: int = 20) -> List[Path]:
    """Optionally poll batches to completion and download outputs/errors."""
    load_dotenv()
    client = OpenAI()
    output_path.mkdir(parents=True, exist_ok=True)
    errors_path = "data/artifacts/api/errors"
    errors_path.mkdir(parents=True, exist_ok=True)
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
            out_path = output_path / f"{bid}_results.jsonl"
            if not out_path.exists():
                stream = client.files.content(out_id)
                with out_path.open("wb") as f:
                    f.write(stream.read())
                print(f"[Download] {bid} -> {out_path}")
                new_outputs.append(out_path)

        err_id = getattr(b, "error_file_id", None)
        if err_id:
            err_path = errors_path / f"{bid}_errors.jsonl"
            if not err_path.exists():
                estream = client.files.content(err_id)
                with err_path.open("wb") as f:
                    f.write(estream.read())
                print(f"[Errors]   {bid} -> {err_path}")

    return 

def check_batch_status(batch_id):
    """
    Checks the status of a batch job.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch = client.batches.retrieve(batch_id)
    return batch.status

# def download_batch_output_file(output, path: str = "output_api_books.jsonl") -> str:
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     stream = client.files.content(output)
#     with open(path, "wb") as f:
#         f.write(stream.read())
#     return path

# def parse_batch_output(path: str = "output_api_books.jsonl"):
#     results = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             row = json.loads(line)
#             custom_id = row.get("custom_id")
#             content = (row.get("response", {})
#                          .get("body", {})
#                          .get("choices", [{}])[0]
#                          .get("message", {})
#                          .get("content", ""))  # model's JSON as a string
#             if not custom_id or not content:
#                 continue
#             results[custom_id] = json.loads(content)  # dict with 4 keys
#     return results
