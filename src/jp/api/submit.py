"""
This file runs the batch inference using the OpenAI API.
"""

from __future__ import annotations

import json
import os

from pathlib        import Path


from dotenv         import load_dotenv
from openai         import OpenAI
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
    
def build_input(case_name: str, 
                inp_path: str = 'data/artifacts/api/requests.jsonl' ):
    """Writes the full input jsonl file for the API from the cases dataframe.
    
    :param books:           Str of books to ask the API
    :param out_path:        Path to write the output jsonl file. 
    """
    inp_path = Path(inp_path)
    inp_path.parent.mkdir(parents=True, exist_ok=True)

    with inp_path.open("w", encoding="utf-8") as f:
        for name in case_name:
            request = BatchRequest(name).to_dict()
            json.dump(request, f, ensure_ascii=False)
            f.write("\n")

book_names = [
    "Harry Potter and the Philosopher's Stone",
    "A Court of Mist and Fury"
]

def request_api(input: str = 'data/artifacts/outputs/api_answers.jsonl'):
    """
    Requests the OpenAI API for a batch job. Every time you call this function, it will ask the API thus cost money. Run ONCE. 
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the JSONL file
    file = client.files.create(
        file=open(input, "rb"),
        purpose="batch"
    )

    # Create the batch job
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print("Batch ID:", batch.id)

    return batch.id

def check_batch_status(batch_id):
    """
    Checks the status of a batch job.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch = client.batches.retrieve(batch_id)
    return batch.status

def download_batch_output_file(output, path: str = "output_api_books.jsonl") -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.files.content(output)
    with open(path, "wb") as f:
        f.write(stream.read())
    return path

def parse_batch_output(path: str = "output_api_books.jsonl"):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            custom_id = row.get("custom_id")
            content = (row.get("response", {})
                         .get("body", {})
                         .get("choices", [{}])[0]
                         .get("message", {})
                         .get("content", ""))  # model's JSON as a string
            if not custom_id or not content:
                continue
            results[custom_id] = json.loads(content)  # dict with 4 keys
    return results

path = download_batch_output_file(b.output_file_id)
data_by_book = parse_batch_output(path)