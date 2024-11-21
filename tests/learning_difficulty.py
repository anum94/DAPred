from ds.supported import load_dataset
from features.Domain import Domain

if __name__ == "__main__":
    domain = "wispermed"
    num_samples = 100
    split = "train"
    dataset = load_dataset(
        dataset=domain,
        samples=num_samples,
    )

    articles = dataset.get_split(split)["text"]
    if len(articles) > num_samples:
        articles = articles[:num_samples]

    d = Domain(articles, domain)
    learn_difficulty = d.compute_learning_difficulty()
    print("LearNING Difficulty:", learn_difficulty)
