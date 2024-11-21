import pandas as pd


df = pd.concat([pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_3_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_12_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_4_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_5_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_6_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_7_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_8_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_9_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_10_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_11_llama3.1_8b_0-shot_500.xlsx"),
                pd.read_excel("training_features/2024-11-14_11-39-15/scores_ds_13_llama3.1_8b_0-shot_500.xlsx"),
                ], axis=0)
df.to_excel("training_features/2024-11-14_11-39-15/scores_llama3.1_8b_0-shot_500.xlsx")
df = df.drop(['Unnamed: 0'], axis=1)
print (len(df))