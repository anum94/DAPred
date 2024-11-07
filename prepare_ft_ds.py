from ds.supported import load_dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub import login
login ("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")
hf_repo_name = {
    "bigpatent": "anumafzal94/bigpatent_10k_finetuning",
    "pubmed": "anumafzal94/pubmed_10k_finetuning",
    "cnndm": "anumafzal94/cnndm_10k_finetuning",
    "samsum": "anumafzal94/samsum_10k_finetuning",
    "billsum": "anumafzal94/billsum10k_finetuning",
    "legalsum": "anumafzal94/legalsum_10k_finetuning",
    "newsroom": "anumafzal94/newsroom_10k_finetuning",
    "aclsum": "anumafzal94/aclsum_10k_finetuning",
    "dialogsum": "anumafzal94/dialogsum_10k_finetuning",
    "gigaword":"anumafzal94/gigaword_10k_finetuning",
    "xlsum": "anumafzal94/xlsum_10k_finetuning",
    "govreport": "anumafzal94/govreport_10k_finetuning",
     "arxiv": "anumafzal94/arxiv_10k_finetuning",
    "wispermed": "anumafzal94/wispermed_10k_finetuning",
}


def add_instruction(sample, _):
    instruction = "You are an expert at summarizing long articles. Proceed to summarize the following text: \n"
    prompt = [instruction + tex for tex in sample['text']]

    return {"text": prompt }
for ds in hf_repo_name.keys():
    dataset = load_dataset(
        dataset=ds,
        samples=10000,
    )
    dataset_train = dataset.get_split("train")
    dataset_train =  dataset_train.map(add_instruction, dataset_train, batched=True)
    train_val_split = dataset_train.train_test_split(test_size=0.1)

    dataset = DatasetDict({'validation': train_val_split['test'],
                           "train": train_val_split['train']})

    dataset.push_to_hub(hf_repo_name[ds])


