# Judge Promotion Project

## Overview
The **Judge Promotion Project** investigates what factors influence the likelihood of a **judge’s promotion**. It combines legal case data from multiple sources and processes it through a structured, six-step data and modeling pipeline.

The core data sources include:
- **CAP (Caselaw Access Project)** – for appellate case opinions  
- **Court Listener (CL)** – for additional case metadata and text  
- **Judges Dataset** – for biographical and career information  

All scripts and helper functions are contained in the `src/jp/` directory.

---

## Project Structure

````text
project_root/
│
├── data/
│   ├── parquet_files/           # Raw input CAP files (12GB total)
│   └── artifacts/               # Generated intermediate datasets
│
├── src/jp/                      # All reusable modules and utilities
├── 01_build_cap_dataset.py
├── 02_scrape_cl.py
├── 03_merge_and_request.py
├── 04_merge_api.py
├── 05_features.py
├── 06_model.py
└── README.md

---
````


## Pipeline Overview

The project runs in **six main steps**, executed sequentially:

### Step 1 — `01_build_cap_dataset.py`
**Goal:** Build the base CAP dataset of appellate cases and match them to lower district judges.

- Reads all chunked CAP parquet files stored in `data/parquet_files/` as `CAP_data_{}.parquet` (total ≈ 12GB).
- Filters cases to the relevant courts.
- Runs `match_appellates()` to link appellate cases to their originating district court cases.
- **Output:** A dataset containing appellate cases and their corresponding lower district judges (with judge IDs), saved under `data/artifacts/cap/`.

---

### Step 2 — `02_scrape_cl.py`
**Goal:** Collect and process data from **Court Listener (CL)**.

- Scrapes case texts from the Court Listener website if files are missing in `data/artifacts/cl/`.
- Filters cases where the opinion text explicitly mentions the lower district judge.
- Integrates this information into the dataset to strengthen judge–case links.
- **Output:** A filtered and enriched CL dataset linking appellate decisions to district judges.

---

### Step 3 — `03_merge_and_request.py`
**Goal:** Merge datasets and classify appellate outcomes.

- Merges the **CAP** and **CL** datasets.
- Prepares API requests to classify each case’s **outcome** (e.g., *affirmed*, *reversed*, *vacated*).
- Uses the **OpenAI API** to extract outcome information directly from the opinion text.
- **Output:** A merged dataset ready for model-based annotation and stored API results.

---

### Step 4 — `04_merge_api.py`
**Goal:** Integrate API responses into the dataset.

- Collects completed API outputs (e.g., via batch runs).
- Merges these results into the main dataset.
- Cleans and validates classification results (affirmed/reversed labels).
- **Output:** A finalized case-level dataset with outcome labels attached.

---

### Step 5 — `05_features.py`
**Goal:** Build judge-level features for modeling.

- Aggregates appellate outcomes at the **judge level**.
- Computes metrics such as:
  - Overturn rate
  - Affirmation ratio
  - Number of appellate reviews per judge
- Combines these metrics with the existing **judges dataset**.
- **Output:** A CSV file with one row per judge and feature columns suitable for modeling.

---

### Step 6 — `06_model.py`
**Goal:** Train the predictive model for judge promotion.

- Loads the judge-level features.
- Preprocesses and encodes relevant variables.
- Trains and evaluates machine learning models to identify promotion predictors.
- **Output:** Model results, evaluation metrics, and visualizations.

---

## How to Run

Run the six scripts **in order**, as each stage builds on the previous one:

```bash
python 01_build_cap_dataset.py
python 02_scrape_cl.py
python 03_merge_and_request.py
python 04_merge_api.py
python 05_features.py
python 06_model.py
```