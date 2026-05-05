from preprocessing import build_tables_for_paper_models, get_full_timeseries_model

if __name__ == "__main__":
    # Build tables for paper models and print out the first few rows of each model's table
    timeseries_model_data = get_full_timeseries_model(build_tables_for_paper_models(build_tables_for_paper_models(open_filtered_standard_data(), return_condition_sets_and_personal_info=True)[0], return_condition_sets_and_personal_info=True)[0])
    