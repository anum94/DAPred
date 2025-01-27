import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("training_features/2024-11-14_11-39-15/scores_llama3.1_8b_0-shot_500.xlsx")
df_all_norm = df.loc[df['features'] == 'all-norm']
df_all_norm = df_all_norm.loc[df_all_norm['feature_selection'] == True]
df_all_norm = df_all_norm.drop(['features'], axis = 1)
df_all_norm.set_index('num_datasets', inplace=True)
df_all_norm.columns = ['DA-Pred_'+col for col in df_all_norm.columns]
#df_all_norm = df_all_norm.where(df_all_norm == 0.11, -1)
#df_all_norm = df_all_norm.where(df_all_norm > 2, 2)



#df_all_raw = df.loc[df['features'] == 'all-raw']
#df_all_raw = df_all_raw.drop(['features'], axis = 1)
#df_all_raw.set_index('num_datasets', inplace=True)
#df_all_raw.columns = [col+'-all' for col in df_all_raw.columns]

df_baseline_norm = df.loc[df['features'] == 'baseline-norm']
df_baseline_norm = df_baseline_norm.loc[df_baseline_norm['feature_selection'] == True]
df_baseline_norm = df_baseline_norm.drop(['features'], axis = 1)
df_baseline_norm.set_index('num_datasets', inplace=True)
df_baseline_norm.columns = ['ROUGE-'+col for col in df_baseline_norm.columns]

#df_baseline_raw = df.loc[df['features'] == 'baseline-raw']
#df_baseline_raw = df_baseline_raw.drop(['features'], axis = 1)
#df_baseline_raw.set_index('num_datasets', inplace=True)
#df_baseline_raw.columns = [col+'-base' for col in df_baseline_raw.columns]

df_plot = pd.concat([df_all_norm,df_baseline_norm], axis = 1)


# Plot Raw vs normalized features
# 1) Baseline
columns_to_plot = [col for col in df_plot.columns if 'r2' in col]
df_plot[columns_to_plot].plot()
plt.legend(fontsize=14)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel("# Datasets",fontsize=18)
plt.ylabel(f"R\N{SUPERSCRIPT TWO}", fontsize=18)
plt.show()

columns_to_plot = [col for col in df_plot.columns if 'rmse' in col]
df_plot[columns_to_plot].plot()
plt.legend(fontsize=14)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel("# Datasets",fontsize=18)
plt.ylabel(f"RMSE", fontsize=18)
plt.show()







