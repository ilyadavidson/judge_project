from    __future__          import annotations
# from    batch_inference     import BatchRequest, batch_request_body
from    pathlib             import Path
from    dotenv              import load_dotenv 
from    not_needed.API_overturnrate    import truncate_opinion, text_builder
from    openai              import OpenAI
from typing import List, Optional, Tuple
from    data_loading        import build_cap_dataset

import glob
import os
import time 
import json

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
        self.model = "gpt-5-mini-2025-08-07"
        self.messages = [
            {"role": "system", "content": system_msg},
            {"role": "developer", "content": developer_msg},
            {"role": "user", "content": user_template.format(opinion_text=opinion_text)},
        ]

    def to_dict(self):
        return {
            "model": self.model,
            "messages": self.messages,
        }

class BatchRequest:
    def __init__(self, custom_id, body: str):
        self.custom_id = str(custom_id)                 # no trailing comma
        self.method = "POST"                            # no trailing comma
        self.url = "/v1/chat/completions"               # no trailing comma
        self.body = batch_request_body(body).to_dict()  # store as dict, not object

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

def run_API_batch_inference(
        input_file:         str = "input_api_format.jsonl",
        max_bytes:          int = 200 * 1024 * 1024,
        output_dir:         str = "input",
        output_prefix:      str = "reponse_part",
        endpoint:           str = "/v1/chat/completions",
        completion_window:  str = "24h",
        work_dir:           str = "batch_runs",
        df:                 any = None,
        ):
    """
    Splits a large JSONL file into roughly equal parts, uploads them to OpenAI for batch processing, returns the API responses.

    :param input_file: Path to the large JSONL file to be split and processed. Has the format required by the OpenAI API, with the messages and query we want to ask the AI.
    :param max_bytes: OpenAI's maximum file size for batch processing is 200MB, so we split the input file into parts no larger than this.
    :param output_dir: Directory to save the split files that will be used as input for the batch API call.
    :param output_prefix: Prefix for the split files.
    :param endpoint: OpenAI API endpoint to use for batch processing.
    :param completion_window: Time window for the batch job to complete.
    :param work_dir: Directory to save intermediate and output files.
    :param df: full df.

    :return: None. Saves the API responses to 'results.jsonl'.
    """
    
    # Set up variables and OpenAI client
    ######################################################################################
    load_dotenv()  
    client          = OpenAI()
    work            = Path(work_dir)
    input_dir       = work / "input_chunks"
    output_dir      = work / "output"

    input_dir.mkdir(parents=True, exist_ok=True) 
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the batch request body
    ######################################################################################
    """
    The API expects a json with a call for every case, this creates that jsonl file.
    """
    cases = text_builder(df, limit=10, max_tokens_each=3000) # should be +- 50.000 for Third Circuit

    res = []
    for index, row in enumerate(cases):
        try:
            res.append(BatchRequest(custom_id=row["id"], body=row["text"]))
        except Exception as e:
            print(f"Error creating BatchRequest for row {index}: {e}")

    with open(input_file, "w", encoding="utf-8") as f:
        for r in res:
            try:
                json.dump(r.to_dict(), f, ensure_ascii=False)
                f.write("\n")
            except Exception as e:
                print(f"Error serializing BatchRequest: {e}")    
    
    # Sanity check
    with open(input_file, "r", encoding="utf-8") as f:
        num_items = sum(1 for _ in f)
    print(f"Number of items in jsonl: {num_items}") # for Third Circuit this should be about 51.000 appellate cases
    
    
    # Split up the input file and upload parts, OpenAI can only take files up to 200MB
    ######################################################################################
    """
    Now that we have the correct input for the API (input_file), we need to split it up in parts no larger than 200MB.
    This returns parth_paths, in the input_dir, which we will upload to OpenAI (batch_runs/input).
    """
    src = Path(input_file)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src.resolve()}")

    part_paths: List[Path] = []
    part_idx               = 1
    bytes_in_part          = 0
    part_path              = input_dir / f"{output_prefix}_{part_idx}.jsonl"
    out_f                  = part_path.open("wb")

    with src.open("rb") as inp:  # read in binary so size is accurate
        for line in inp:
            line_len = len(line)

            if bytes_in_part > 0 and bytes_in_part + line_len > max_bytes:
                out_f.close()
                print(f"[Split] Finalized {part_path} at {bytes_in_part} bytes")

                part_idx        += 1
                bytes_in_part   = 0
                part_path       = input_dir / f"{output_prefix}_{part_idx}.jsonl"
                out_f           = part_path.open("wb")

            out_f.write(line)
            bytes_in_part       += line_len

    out_f.close()
    part_paths = sorted(input_dir.glob(f"{output_prefix}_*.jsonl"))

    if not part_paths:
        raise RuntimeError("No parts were created — nothing to upload.")

    # Upload all the parts to OpenAI
    # #####################################################################################
    uploaded_file_ids: List[str] = []
    for p in part_paths:
        with p.open("rb") as fh:
            file_obj = client.files.create(file=fh, purpose="batch")
        uploaded_file_ids.append(file_obj.id)

    # Create a batch request for each uploaded file
    ######################################################################################
    batch_ids: List[str] = []
    for file_id in uploaded_file_ids:
        batch = client.batches.create(
            input_file_id       =   file_id,
            endpoint            =   endpoint,
            completion_window   =   completion_window,
            metadata            =   {"description": f"Batch for {file_id}"},
        )
        batch_ids.append(batch.id)
        print(f"[Batch] Created batch {batch.id} for input file {file_id}")

    # Download all if done
    ######################################################################################
    result_part_files: List[Path] = []
    error_part_files: List[Path] = []
    terminal_statuses = {"completed", "failed", "cancelled", "expired"}

    for bid in batch_ids:
        # Poll until the batch reaches a terminal state
        while True:
            b = client.batches.retrieve(bid)
            status = getattr(b, "status", None)
            print(f"[Poll] batch_id={bid} status={status}")
            if status in terminal_statuses:
                break
            time.sleep(10)

        # Download output if present
        out_fid = getattr(b, "output_file_id", None)
        if out_fid:
            resp = client.files.content(out_fid)
            part_out = output_dir / f"{bid}_results.jsonl"
            with part_out.open("wb") as fh:
                fh.write(resp.read())
            print(f"[Download] output batch_id={bid} -> {part_out}")
            result_part_files.append(part_out)
        else:
            print(f"[Info] batch_id={bid} has no output_file_id")

        # Download errors if present
        err_fid = getattr(b, "error_file_id", None)
        if err_fid:
            err_stream = client.files.content(err_fid)
            part_err = output_dir / f"{bid}_errors.jsonl"
            with part_err.open("wb") as fh:
                fh.write(err_stream.read())
            print(f"[Download] errors batch_id={bid} -> {part_err}")
            error_part_files.append(part_err)

    if not result_part_files:
        raise RuntimeError("No result files were downloaded.")

    # Merge everything in one output file
    ######################################################################################
    merged_path = work / "results_merged.jsonl"
    with merged_path.open("wb") as out:
        for rf in result_part_files:
            with rf.open("rb") as inp:
                out.write(inp.read())
    print(f"[Merge] Merged {len(result_part_files)} file(s) into {merged_path.resolve()}")

    return merged_path

def download_and_merge_batch_outputs(
    work_dir: str | Path = "batch_runs",
    *,
    include_errors: bool = True,
    only_completed: bool = True,
) -> Tuple[List[Path], Optional[Path]]:
    """
    Lists batches, downloads output/error files for completed ones, and merges outputs.

    Behavior:
      - Skips re-downloading any per-batch output/error files that already exist.
      - If `results_merged_resume.jsonl` DOES NOT exist: build it from ALL existing per-batch outputs.
      - If it DOES exist: APPEND ONLY newly downloaded outputs to it.

    Returns:
        (newly_downloaded_outputs, merged_path)
    """
    load_dotenv()
    client = OpenAI()

    work_dir = Path(work_dir)
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = client.batches.list()
    print("Found", len(batches.data), "batches")

    newly_downloaded: List[Path] = []

    for b in batches.data:
        if only_completed and getattr(b, "status", None) != "completed":
            continue

        bid = b.id
        out_id = getattr(b, "output_file_id", None)
        err_id = getattr(b, "error_file_id", None)

        if out_id:
            out_path = out_dir / f"{bid}_results.jsonl"
            if out_path.exists():
                print(f"[Skip] Output for {bid} already exists at {out_path}")
            else:
                stream = client.files.content(out_id)
                with out_path.open("wb") as f:
                    f.write(stream.read())
                print(f"[Download] {bid} -> {out_path}")
                newly_downloaded.append(out_path)

        if include_errors and err_id:
            err_path = out_dir / f"{bid}_errors.jsonl"
            if err_path.exists():
                print(f"[Skip] Error file for {bid} already exists at {err_path}")
            else:
                estream = client.files.content(err_id)
                with err_path.open("wb") as f:
                    f.write(estream.read())
                print(f"[Errors]   {bid} -> {err_path}")

    merged = work_dir / "results_merged_resume.jsonl"

    # Decide how to (re)build the merged file
    if merged.exists():
        if newly_downloaded:
            # Append only new outputs
            with merged.open("ab") as out_f:
                for p in newly_downloaded:
                    with p.open("rb") as inp:
                        out_f.write(inp.read())
            print(f"[Append] Added {len(newly_downloaded)} new file(s) to {merged}")
        else:
            print("[Append] No new outputs to append; merged file unchanged.")
    else:
        # Build from ALL existing outputs if merged doesn't exist yet
        all_outputs = sorted(out_dir.glob("*_results.jsonl"))
        if not all_outputs:
            print("No completed batches with output to download or merge yet.")
            return newly_downloaded, None
        with merged.open("wb") as out_f:
            for p in all_outputs:
                with p.open("rb") as inp:
                    out_f.write(inp.read())
        print(f"[Merge] Built merged file from {len(all_outputs)} outputs -> {merged}")

    return newly_downloaded, merged

if __name__ == "__main__":
    # df = build_cap_dataset()
    # final_path = run_API_batch_inference(
    #     input_file="input_api_format.jsonl",
    #     max_bytes=200 * 1024 * 1024,
    #     output_dir="input",
    #     output_prefix="response_part",
    #     endpoint="/v1/chat/completions",       
    #     completion_window="24h",
    #     work_dir="batch_runs",
    #     df=df

    newly_downloaded, merged = download_and_merge_batch_outputs(work_dir="batch_runs", include_errors=True, only_completed=True)