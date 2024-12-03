# Imports
import os.path
from functools import lru_cache

import nltk
import numpy
from tqdm import tqdm

nltk.download("wordnet")
import logging
import random
from datetime import datetime
from typing import List

import pandas as pd

from ds.supported import load_dataset
from features.Domain import Domain
from features.Similarity import Similarity


def get_task_spec_metrics(domain: str, task: str, task_spec_metrics):
    task_spec_metrics = task_spec_metrics.drop(["ds", "split", "model"], axis=1)
    if task == "classification":
        return {"accuracy": random.uniform(0, 1)}
    elif task == "summarization":
        if len(task_spec_metrics) > 0:
            score_dict = task_spec_metrics.iloc[0].to_dict()
        else:  # return random values
            logging.warning(f"Added random value for task: {task}, domain:{domain} \n")
            score_dict = dict()
            for score in task_spec_metrics.columns:
                score_dict[score] = random.uniform(0, 1)
        # @todo: drop the non-relavant features here so they are not included in y_weighted.
        return score_dict
    else:
        return {}


def get_domain_specific_metrics(domain: str, num_samples=100):
    d = get_domain(domain=domain, split="test",num_samples=num_samples )
    return {"learning_difficult": d.compute_learning_difficulty()}


@lru_cache(maxsize=32)
def get_domain(domain, split, num_samples=5):
    dataset = load_dataset(
        dataset=domain,
        samples=num_samples,
    )

    articles = dataset.get_split(split)["text"]
    if len(articles) > num_samples:
        articles = articles[:num_samples]

    d = Domain(articles, domain)
    return d


def get_domain_similarity_metrics(source: str, target: str, da: str, num_samples=100):
    if source == target and da == "in-domain-adapt":
        source_split = "train"
        target_split = "test"
    elif source == target and da == "no-domain-adapt":
        source_split = "train"
        target_split = "train"
    else:
        source_split = "test"
        target_split = "test"

    # start = time.time()
    S = get_domain(domain=source, split=source_split, num_samples=num_samples)
    # end = time.time()
    # print(f"{source} domain computation took {end - start}")

    # start = time.time()
    T = get_domain(domain=target, split=target_split, num_samples=num_samples)
    # end = time.time()
    # print(f"{source} domain computation took {end - start}")

    ST = Similarity(S, T)

    return {
        "vocab-overlap": ST.vocab_overlap,
        "tf-idf-overlap": ST.tf_idf_overlap,
        "source_shannon_entropy": S.shannon_entropy,
        "target_shannon_entropy": T.shannon_entropy,
        "kl-divergence": ST.kl_divergence,
        "js-divergence": ST.js_divergence,
        "contextual-overlap": ST.contextual_overlap,
    }


def weighted_average(nums, weights):
    return sum(x * y for x, y in zip(nums, weights)) / sum(weights)


def get_features(
    da: str, source: str, target: str, task: str, task_scores, num_samples, ft = False
) -> (List, List):
    features = []
    feature_names = [
        "da-type",
        "source",
        "target",
        "ft"
    ]
    features.append(da)
    features.append(source)
    features.append(target)
    features.append(ft)

    domain_spec_features = get_domain_specific_metrics(target, num_samples=num_samples)
    features += list(domain_spec_features.values())
    feature_names += list(domain_spec_features.keys())

    domain_similarity_features = get_domain_similarity_metrics(
        target, source, da, num_samples=num_samples
    )
    features += list(domain_similarity_features.values())
    feature_names += list(domain_similarity_features.keys())

    try:
        source_model = "meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo"
        target_model = "meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo"
        if source == target and da == "in-domain-adapt":
            target_split = "test"
            source_split = "test"
            if ft:
                source_model = "meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo"
                target_model = "anumafzal94-llama3.1"

        elif source == target and da == "no-domain-adapt":
            target_split = "test"
            source_split = "train"
            if ft:
                source_model = "anumafzal94-llama3.1"
                target_model = "anumafzal94-llama3.1"
                source_split = "test"

        else:
            target_split = "test"
            source_split = "test"
            if ft:
                source_model = "anumafzal94-llama3.1"
                target_model = "anumafzal94-llama3.1"

        source_task_scores = task_scores.loc[
            (task_scores["ds"] == source) & (task_scores["split"] == source_split) & (task_scores["model"] == source_model)
        ]
    except:
        # todo: add a dummy variable for this
        print(
            f"Failed to get scores for domain {source} for {da} setting. Assigning dummy scores."
        )

    task_specific_feature = get_task_spec_metrics(source, task, source_task_scores)
    features += list(task_specific_feature.values())
    feature_names += [f"source_{key}" for key in list(task_specific_feature.keys())]
    feature_weight = [1 / len(task_specific_feature.values())] * len(task_specific_feature.values())  # equal weight to all features
    weighted_y_source = weighted_average(
        list(task_specific_feature.values()), feature_weight
    )

    target_task_scores = task_scores.loc[(task_scores["ds"] == target) & (task_scores["split"] == target_split) & (task_scores["model"] == target_model)]
    task_specific_feature = get_task_spec_metrics(target, task, target_task_scores)
    features += list(task_specific_feature.values())
    feature_names += [f"target_{key}" for key in list(task_specific_feature.keys())]
    feature_weight = [1 / len(task_specific_feature.values())] * len(task_specific_feature.values())  # equal weight to all features
    weighted_y_target = weighted_average(
        list(task_specific_feature.values()), feature_weight
    )

    y_drop = weighted_y_source - weighted_y_target

    features += [weighted_y_target, weighted_y_source, y_drop]
    feature_names += ["y_weighted_source", "y_weighted_target", "y_drop"]

    return features, feature_names


def get_template(task_scores: pd.DataFrame, num_domains=None, num_samples=10, ft = False) -> pd.DataFrame:


    domains = list(set(task_scores["ds"]))
    domains = ['arxiv', 'gigaword', 'wispermed', 'govreport']
    #if num_domains is not None:
    #    if len(domains) > num_domains:
    #        domains = domains[:num_domains]
    #task_scores = task_scores.drop(columns=task_scores.columns[:19])
    da_type = ["in-domain-adapt", "single-domain-adapt", "no-domain-adapt"]
    task = "summarization"
    feature_names = ["dummy_feature_name"] * (len(task_scores.columns) - 1)
    df = pd.DataFrame()

    for da in tqdm(da_type):
        features = []
        features.append(da)
        if da == "in-domain-adapt" or da == "no-domain-adapt":
            for domain in tqdm(domains):
                features, feature_names = get_features(
                    da, domain, domain, task, task_scores, num_samples, ft=ft
                )
                if df.columns.empty:
                    df = pd.DataFrame(columns=feature_names)
                df.loc[len(df)] = features

        elif da == "single-domain-adapt":
            for source in tqdm(domains):
                domains_copy = domains.copy()
                domains_copy.remove(source)
                for target in domains_copy:
                    features, feature_names = get_features(
                        da, source, target, task, task_scores, num_samples, ft = ft
                    )
                    if df.columns.empty:
                        df = pd.DataFrame(columns=feature_names)
                    df.loc[len(df)] = features
        else:
            df.loc[len(df)] = [numpy.NaN for i in range(len(feature_names))]
    # clear_cache()
    write_logs(df)

    return df


def write_logs(df: pd.DataFrame):
    date_time = "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.now())
    folder = os.path.join("logs", date_time)
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, "features.csv"), index=False)

    print(f"Run logs would be locally stored at {folder}")


def construct_training_corpus(
    num_domains: int,
    num_samples,
    da_type: str = "in-domain-adapt",
    template_path: str = "template.xlsx",
) -> pd.DataFrame:
    assert da_type in ["in-domain-adapt", "single-domain-adapt"]

    df = pd.read_excel(template_path, header=0)
    df_zero_shot = df.loc[df['model'] == 'meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo']
    df_ft = df.loc[df['split'] == 'test']

    template_2 = get_template(df_ft, num_domains=num_domains, num_samples=num_samples, ft = True
                              )
    template_1 = get_template(df_zero_shot, num_domains=num_domains, num_samples=num_samples
    )



    # print (template)
    template = pd.concat([template_1, template_2], axis = 0)
    return template
