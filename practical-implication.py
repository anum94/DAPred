import pandas as pd
from features.features import get_template_lr
from dotenv import load_dotenv
print(load_dotenv())
# Read the dataset
#df_legal = pd.read_csv("low-resource_datasets/MODEL_meta-llama-Meta-Llama-3.1-8B-Instruct-Turbo_0-SHOT_unseen_legal_data_test_2025-09-12_10-22-38.csv")
#df_medical = pd.read_csv("low-resource_datasets/unseen_medical_data.csv")
#df_news = pd.read_csv("low-resource_datasets/unseen_news_data.csv")
#df_scientific = pd.read_csv("low-resource_datasets/unseen_scientific_data.csv")



# Find the most similar Domain(s). One or more datasets belonging to one or more domain


# Compute features on the fly
df1 = pd.read_excel("low-resource_datasets/overall_summary.xlsx", usecols=lambda c: not str(c).startswith('Unnamed'))
df1 = df1[df1['model_hf_key'].str.contains('Llama', na=False)]
df2 = pd.read_csv("low-resource_datasets/overall_summary_ds_14_llama3.1_8b_zeroshot.csv", usecols=lambda c: not str(c).startswith('Unnamed'))
df = pd.concat([df1, df2], ignore_index=True)
df.drop('fs_wiki', axis=1, inplace=True)
df = df[df['split'].str.contains('test', na=False)]

#domain similarity calculation when source is "law"
df_law = df[~df['dataset_name'].isin(['medical', 'scientific', 'news'])]
features = get_template_lr(df_law, lr_domain="legal", num_samples=2, ft = False)
# Forward pass on the regression model and take average if more than one

