
# 📉 DA-Pred: Performance Prediction for Text Summarization

[![Paper](https://img.shields.io/badge/ACL_Anthology-2025.emnlp--main.387-B31B1B.svg)](https://aclanthology.org/2025.emnlp-main.387/)
[![Conference](https://img.shields.io/badge/EMNLP-2025-blue)](https://2025.emnlp.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](#)

Official repository for the paper: **"DA-Pred: Performance Prediction for Text Summarization under Domain-Shift and Instruct-Tuning"**, published in *EMNLP 2025*.

**DA-Pred** is a lightweight framework that predicts the **Performance Drop** of Large Language Models (LLMs) when transitioning from a high-resource source domain (e.g., News) to a low-resource target domain (e.g., Legal, Medical). It allows practitioners to estimate how much a model's ROUGE or BERTScore will decline under domain-shift without requiring any labeled data for the target domain.

---

## 🛠 Setup & Data Requirements

### 1. Installation
```bash
git clone [https://github.com/anum94/DAPred.git](https://github.com/anum94/DAPred.git)
cd DAPred
pip install -r requirements.txt
```

### 2. Data Templates
The framework uses two primary Excel files to store and process performance scores. Before running the pipeline, ensure these are populated in the root directory:

* **`template1.xlsx`**: Stores the **Zero-Shot** performance scores (Source and Target domains).
* **`template2_ft.xlsx`**: Stores the performance scores of **Fine-Tuned** models.

---

## 🚀 Running the Code

The `main.py` script executes the complete prediction pipeline: loading experimental results, calculating the performance delta, and training regression models to predict those drops for unseen settings.

### Execution
```bash
python main.py
```

### How it Works
1.  **Metric Aggregation:** The script reads performance metrics (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore) from your templates.
2.  **Performance Delta Calculation:** It calculates the shift in performance ($\Delta$) between high-resource source domains and low-resource target domains.
3.  **Regression Modeling:** It trains and evaluates various regression models (e.g., Random Forest, Linear Regression) to learn how domain-specific features (like vocabulary overlap and source performance) predict the eventual performance drop.
4.  **Prediction:** The trained predictor can then estimate the expected drop for a new model or a new target domain where labels are unavailable.

---

## 📊 Methodology Overview

DA-Pred addresses the challenge of evaluating LLMs in the "Wild." Instead of annotating thousands of new summaries, we treat performance prediction as a regression task based on:
* **Known Performance:** How well the model performs on high-resource data.
* **Domain Divergence:** Quantitative distance between the source and target domains (e.g., vocabulary overlap).
* **Model Type:** Accounting for shifts between base models and instruct-tuned variants.

---

## ✍️ Citation

If you use this code or our methodology, please cite:

```bibtex
@inproceedings{afzal-etal-2025-da,
    title = "{DA}-Pred: Performance Prediction for Text Summarization under Domain-Shift and Instruct-Tuning",
    author = "Afzal, Anum and Matthes, Florian and Fabbri, Alexander",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2025.emnlp-main.387](https://aclanthology.org/2025.emnlp-main.387)"
}
```

---
**Authors:** Anum Afzal, Florian Matthes, Alexander Fabbri.
```