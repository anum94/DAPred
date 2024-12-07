import pandas as pd
import matplotlib.pyplot as plt


#df = pd.read_excel("training_features/2024-11-21_15-40-46/scores_llama3.1_8b_0-shot_100.xlsx")
df=pd.read_excel("training_features/2024-12-06_00-04-14/scores_llama3.1_8b_0-shot_500_gpu.xlsx")
df_all_norm = df.loc[df['features'] == 'all-norm']
df_all_norm = df_all_norm.drop(['features'], axis = 1)
df_all_norm.set_index('num_datasets', inplace=True)
df_all_norm.columns = [col+'-all-norm' for col in df_all_norm.columns]
#df_all_norm = df_all_norm.where(df_all_norm == 0.11, -1)
#df_all_norm = df_all_norm.where(df_all_norm > 2, 2)



#df_all_raw = df.loc[df['features'] == 'all-raw']
#df_all_raw = df_all_raw.drop(['features'], axis = 1)
#df_all_raw.set_index('num_datasets', inplace=True)
#df_all_raw.columns = [col+'-all' for col in df_all_raw.columns]

df_baseline_norm = df.loc[df['features'] == 'baseline-norm']
df_baseline_norm = df_baseline_norm.drop(['features'], axis = 1)
df_baseline_norm.set_index('num_datasets', inplace=True)
df_baseline_norm.columns = [col+'-base-norm' for col in df_baseline_norm.columns]

#df_baseline_raw = df.loc[df['features'] == 'baseline-raw']
#df_baseline_raw = df_baseline_raw.drop(['features'], axis = 1)
#df_baseline_raw.set_index('num_datasets', inplace=True)
#df_baseline_raw.columns = [col+'-base' for col in df_baseline_raw.columns]

df_plot = pd.concat([df_all_norm,df_baseline_norm], axis = 1)


# Plot Raw vs normalized features
# 1) Baseline
columns_to_plot = [col for col in df_plot.columns if 'base' in col and 'r2' in col]
df_plot[columns_to_plot].plot()
plt.show()

columns_to_plot = [col for col in df_plot.columns if 'base' in col and 'rmse' in col]
df_plot[columns_to_plot].plot()
plt.show()

#2) All features
columns_to_plot = [col for col in df_plot.columns if 'all' in col and 'r2' in col]
df_plot[columns_to_plot].plot()
plt.show()

columns_to_plot = [col for col in df_plot.columns if 'all' in col and 'rmse' in col]
df_plot[columns_to_plot].plot()
plt.show()



# Plot of r2
columns_to_plot = [col for col in df_plot.columns if 'r2' in col]
df_plot[columns_to_plot].plot()
plt.show()

# Plot of rmse
columns_to_plot = [col for col in df_plot.columns if 'rmse' in col]
df_plot[columns_to_plot].plot()
plt.show()


# Plot of r2
columns_to_plot = [col for col in df_plot.columns if '-mse-' in col]
df_plot[columns_to_plot].plot()
plt.show()

# Plot of r2
columns_to_plot = [col for col in df_plot.columns if '-mae-' in col]
df_plot[columns_to_plot].plot()
plt.show()








