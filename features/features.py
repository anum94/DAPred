# Imports
import os.path
import re
import json
import time
import random

import numpy
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse
import nltk
nltk.download('wordnet')
from features.Domain import Domain
from features.Similarity import Similarity
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from ds.supported import load_dataset
from typing import List
import pandas as pd
from datetime import datetime
import random
import logging
import itertools
# Definitions



def get_task_spec_metrics(domain:str, task:str, task_spec_metrics):
    task_spec_metrics = task_spec_metrics.drop(["dataset_name"], axis = 1)
    if task == "classification":
        return {"accuracy": random.uniform(0, 1)}
    elif task == "summarization":
        if len(task_spec_metrics) > 0:
            score_dict = task_spec_metrics.iloc[0].to_dict()
        else: #return random values
            logging.warning(f"Added random value for task: {task}, domain:{domain} \n")
            score_dict = dict()
            for score in task_spec_metrics.columns:
                score_dict[score] = random.uniform(0, 1)

        return score_dict
    else:
        return {}

def get_domain_specific_metrics(domain:str):
    return {"learning_difficult": random.uniform(0, 1)}


def get_domain_similarity_metrics(source:str, target:str, num_samples = 100):

    s_dataset = load_dataset(
            dataset=source,
            samples=num_samples,
        )
    s_dval = s_dataset.get_split("train")['text']

    t_dataset = load_dataset(
        dataset=target,
        samples=num_samples,
    )
    t_dval = t_dataset.get_split("train")['text']

    client = OpenAI()

    S = Domain(s_dval, source, client)
    T = Domain(t_dval, target, client) #, unique=True)
    ST = Similarity(S, T)

    return {"word-overlap": ST.word_overlap,
    "vocab-overlap": ST.vocab_overlap,
    "relevance-overlap": ST.relevance_overlap,
    "renyi-divergence": ST.renyi_divergence,
    "kl-divergence": ST.kl_divergence,
    "js-divergence": ST.js_divergence,
    }

def weighted_average(nums, weights):
  return sum(x * y for x, y in zip(nums, weights)) / sum(weights)
def get_features( da:str,source:str, target:str, task:str, task_scores)-> (List,List):
    features = []
    feature_names = ['da-type', 'source', 'target',]
    features.append(da)
    features.append(source)
    features.append(target)

    domain_spec_features = get_domain_specific_metrics(target)
    features += list(domain_spec_features.values())
    feature_names += list(domain_spec_features.keys())

    domain_similarity_features = get_domain_similarity_metrics(target, source, num_samples = 20)
    features += list(domain_similarity_features.values())
    feature_names += list(domain_similarity_features.keys())

    source_task_scores = task_scores.loc[task_scores['dataset_name'] == source]
    task_specific_feature = get_task_spec_metrics(source, task, source_task_scores)
    features += list(task_specific_feature.values())
    feature_names += [f'source_{key}' for key in list(task_specific_feature.keys())]
    feature_weight = [1/len(task_specific_feature.values())] * len(task_specific_feature.values()) #equal weight to all features
    weighted_y_source = weighted_average(list(task_specific_feature.values()), feature_weight)

    target_task_scores = task_scores.loc[task_scores['dataset_name'] == target]
    task_specific_feature = get_task_spec_metrics(target, task, target_task_scores)
    features += list(task_specific_feature.values())
    feature_names += [f'target_{key}' for key in list(task_specific_feature.keys())]
    feature_weight = [1 / len(task_specific_feature.values())] * len(task_specific_feature.values())  # equal weight to all features
    weighted_y_target = weighted_average(list(task_specific_feature.values()), feature_weight)


    y_drop = weighted_y_source - weighted_y_target

    features += [weighted_y_target, weighted_y_source, y_drop]
    feature_names += ['y_weighted_source','y_weighted_target','y_drop']


    return features, feature_names

def get_template(scores_path:str, domains, ) -> pd.DataFrame:

    task_scores = pd.read_excel(scores_path,)
    ds_list = list(task_scores["dataset_name"])
    task_scores = task_scores.drop(columns=task_scores.columns[:15])
    task_scores["dataset_name"] = ds_list
    da_type = ["in-domain-adapt", "single-domain-adapt"] #, "multi-domain-adapt"]
    task = 'summarization'
    feature_names = ['dummy_feature_name'] * (len(task_scores.columns) - 1)
    df = pd.DataFrame()

    for da in da_type:
        features = []
        features.append(da)
        if da == "in-domain-adapt":
            # todo: for in-domain, take train split for source and test split for target
            for domain in domains:
                features, feature_names = get_features(da,domain,domain, task, task_scores)
                if df.columns.empty:
                    df = pd.DataFrame(columns=feature_names)
                df.loc[len(df)] = features

        elif da == "single-domain-adapt":
            for source in domains:
                domains_copy = domains.copy()
                domains_copy.remove(source)
                for target in domains_copy:
                    features, feature_names = get_features(da, source, target, task, task_scores)
                    if df.columns.empty:
                        df = pd.DataFrame(columns=feature_names)
                    df.loc[len(df)] = features
        else:
            df.loc[len(df)] = [numpy.NaN for i in range(len(feature_names))]

    write_logs(df)
    return df
def write_logs(df:pd.DataFrame):
    date_time =  '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.now() )
    folder = os.path.join("logs",date_time)
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder,"features.csv"), index=False)

    print(f"Run logs would be locally stored at {folder}")
def construct_training_corpus(domains: List, da_type: str = "in-domain-adapt",
                              template_path: str = "template.xlsx") -> pd.DataFrame:

    assert da_type in ["in-domain-adapt", "single-domain-adapt", "multi-domain-adapt"]

    template = get_template(template_path, domains = domains)
    print (template)
    return template


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--da_type',
                        type=str,
                        default="in-domain-adapt")
    parser.add_argument('--domain',
                        dest="domains",
                        action='append',
                        default = ['arxiv', 'pubmed', 'govreport', 'wispermed'])

    parser.add_argument('--template_path',
                        type=str,
                        default="template.xlsx")

    args = parser.parse_args()
    construct_training_corpus(domains=args.domains, da_type=args.da_type,template_path=args.template_path)



