def run_cl_extracted_incremental():
    """
    1) Read cl_extracted.csv
    2) Append any NEW ids to overlap_input2.jsonl (requests file)
    3) Batch only ids NOT YET answered in cl_scrape_results.jsonl
    4) Append new results into cl_scrape_results.jsonl
    """
    # --- load source
    cl_extracted = pd.read_csv("cl_extracted.csv")

    # --- how to compute custom_id from each row
    id_getter = lambda i, r: str(r.get("unique_id", "")).replace("CL_", "", 1) or str(i)

    # --- ensure requests file contains all df ids (append only new)
    appended_reqs = append_new_requests(
        df=cl_extracted,
        input_jsonl="batch_runs/overlap_input2.jsonl",
        build_request_fn=build_request_fn,
        id_getter=id_getter,
    )
    print(f"[Requests] Appended {appended_reqs} new requests into overlap_input2.jsonl")

    # --- figure out which ids still need answers
    df_with_ids = _df_with_ids(cl_extracted, id_getter)
    all_ids_in_df   = set(df_with_ids["custom_id"].astype(str))
    answered_ids    = _existing_custom_ids("cl_scrape_results.jsonl")
    ids_to_query    = sorted(all_ids_in_df - answered_ids)

    if not ids_to_query:
        print("[Skip] No new ids to ask the API (everything already in cl_scrape_results.jsonl).")
        return

    # --- build a *temporary* JSONL containing only the missing ids
    missing_df = df_with_ids[df_with_ids["custom_id"].isin(ids_to_query)]
    tmp_missing_path = Path("batch_runs/overlap_input2_missing.jsonl")
    n_missing = _write_jsonl_for_ids(missing_df, tmp_missing_path, id_getter, build_request_fn)
    print(f"[Build] Missing-only: {n_missing} -> {tmp_missing_path}")

    # --- enqueue this missing-only file, poll, download
    batch_id = enqueue_single_file(tmp_missing_path, ENDPOINT, COMPLETION_WINDOW)
    res_path, err_path = download_outputs_for_batch(batch_id, OUTPUT_DIR, poll=POLL, poll_interval=POLL_INTERVAL)

    # --- merge downloaded results into the master results file (dedup by custom_id)
    if res_path is not None:
        added = append_results_unique(
            incoming_results_jsonl=res_path,
            master_results_jsonl="cl_scrape_results.jsonl",
        )
        print(f"[Results] Appended {added} new records into cl_scrape_results.jsonl")
    else:
        print("[Warn] No results file produced for this batch yet.")