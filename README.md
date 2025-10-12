## Judge Promotion Project
# Overview
This project aims to see what drives judge promotions. This is done in 6 steps, which can be found in scripts. The data comes from three datasets, CAP and Court Listener (CL) for the cases and a judges dataset that contains information about judges. 

# 01_build_cap_dataset
The data should be put in data/parquet_files/ and set up as CAP_data_{}.parrquet as the total is 12GB. We read it in chunks. 
This python file reads the chunks, filters on courts and runs match_appellates which is a script that tries to find which district judge's opinion a certain appellate case is ruling over. 
You'll end with a dataset with appellate cases and their corresponding lower district judge with judge id. 

# 02_scrape_cl
This python file scrapes the Court Listener website if the file is not existing in your repo (should be in data/artifacts/cl/{}.csv). If it already exists it continues with filtering the cases based on it having the lower district judge name in the text and then adds this info to the dataset.
You'll end with a dataset with appellate cases and their corresponding lower district judge with judge id. 

# 03_merge_and_request
This python file merges the CL and CAP dataset and builds a request for the OpenAI API to classify information out of the opinion_text of these cases, most importanly what the opinion was (affirmed/reversed etc.)

# 04_merge_api
Once we have the answers, this file fetches the responses and adds the answers to the files.

# 05_features
This file makes the features based off the answers we have for every judge, so based on the cases file that we've been using so far, we can calculate the overturnrate for every judge in the judges file and end up with a csv with every judge and features.

# 06_model
This file is the final model. 

To run, run all these files back-to-back. 
All functions needed for these files are in src/jp. 