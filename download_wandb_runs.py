import json
import os
import shutil
from time import sleep
from typing import List, Optional, Tuple

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

def load_wandb_tables_by_project(entity: str, project: str, out_dir: str = "inference_results", selected_run_ids: Optional[List[str]] = None):
    # initialize API client
    api = wandb.Api()
    failed_runs = []

    # make sure the directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get all runs in the project
    runs = api.runs(path=f"{entity}/{project}")
    df_all_ds = None
    for run in runs:
        try:
            if selected_run_ids and run.id not in selected_run_ids:
                continue
            if run.state != 'finished':
                continue
            table_names = ["rogue_table", "llm_eval_table", "bertscore_table", "FactScore", "vocab_overlap_table"]
            json_config = json.loads(run.json_config)
            print (run.name)
            model_name = run.name.split('_')[1]

            ds_name = run.name.split('_')[3]
            split = run.name.split('_')[-1]

            table_artifact = run.logged_artifacts()

            df_all_scores = None
            for tab_art in table_artifact:
                for table_name in table_names:
                    if table_name in tab_art.name:
                        table_dir = tab_art.download()
                        table_path = f"{table_dir}/{table_name}.table.json"
                        with open(table_path) as file:
                            json_dict = json.load(file)
                        df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
                        if df_all_scores is None:
                            df_all_scores = df
                        else:
                            df_all_scores = pd.concat([df_all_scores,df], axis = 1)
            df_all_scores = df_all_scores.drop(['model',  'hashcode'], axis = 1)
            df_all_scores['model'] = [model_name]
            df_all_scores['ds'] = [ds_name]
            df_all_scores['split'] = [split]
            print(f"Processed run {run.id}")
        except Exception as e:
            print (e)
            print (f"problem with {run.id} {run.name}")
            failed_runs.append(run.id)
        if df_all_ds is None:
            df_all_ds = df_all_scores
        else:
            df_all_ds = pd.concat([df_all_ds, df_all_scores], axis=0)
    out_path = os.path.join(out_dir, "inference_results_ds_13_500_all.xlsx")
    print (out_path)
    df_all_ds.to_excel(out_path, index=False)

    df_zero_shot = df_all_ds.loc[df_all_ds['model'] == 'meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo']
    out_path = os.path.join(out_dir, "inference_results_ds_13_500_0-shot.xlsx")
    print(out_path)
    df_zero_shot.to_excel(out_path, index=False)

    df_ft = df_all_ds.loc[df_all_ds['split'] == 'test']
    out_path = os.path.join(out_dir, "inference_results_ds_13_500_ft.xlsx")
    print(out_path)
    df_ft.to_excel(out_path, index=False)

    sleep(1)
    artifacts_root_dir = "artifacts"
    shutil.rmtree(artifacts_root_dir)
    return failed_runs


def load_wandb_domain_adaptation_summarization_data( entity: str, project:str, selected_run_ids: Optional[List[str]] = None):


    return load_wandb_tables_by_project(entity=entity, project=project, selected_run_ids=selected_run_ids)



def main():
    entity = "anum-afzal-technical-university-of-munich"

    project = "domain-adaptation-summarization"

    failed_runs = load_wandb_domain_adaptation_summarization_data( entity=entity, project=project)
    print (failed_runs)

if __name__ == "__main__":
    main()