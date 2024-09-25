# da-taskagnostic

A task-agnostic score for performance prediction of models in general NLP tasks. See notes below for quick installation and usage.


### install
- Use requirements.txt
```
pip install -r requirements.txt
```

### usage notes
- Create a .env file with  ```export OPENAI_API_KEY=""```
- To download the necessary datasets, specify the noted parameters in `config.yaml`. Then, run


- See TODOs in code for further development and comments about using OpenAI embeddings for some calculations
- To predict performance drop, the current features are as follows:
```
js divergence,
kl divergence,
renyi divergence,
word overlap,
vocab overlap,
relevance overlap,
sentence overlap,
summarization source domain performance data from Anum (BERT, DVO, ROUGE)
classification source domain performance data (need to be implemented, see TODO)
```
