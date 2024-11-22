import argparse
import os.path
import gc
from os import mkdir
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from datetime import datetime
import numpy as np
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
import functools
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from features.features import construct_training_corpus

baseline_feature_target = ['target_rouge1', 'target_rouge2', 'target_rougeL',
                           'target_vocab_overlap']

baseline_feature_source = ['source_rouge1', 'source_rouge2', 'source_rougeL',
                           'source_vocab_overlap']

domain_specific_features = ['learning_difficult', 'vocab-overlap',
                             'kl-divergence', 'js-divergence',
                             'tf-idf-overlap',
                            'source_shannon_entropy','target_shannon_entropy'
                            ]
features_to_normalize = {'source': ['source_bert_precision', 'source_bert_recall', 'source_bert_f1', 'source_vocab_overlap',
                            'source_Relevance', 'source_Coherence', 'source_Consistency', 'source_Fluency'],
                         'all': ['source_shannon_entropy','target_shannon_entropy', 'kl-divergence', 'js-divergence',
                                 'vocab-overlap', 'tf-idf-overlap' , 'learning_difficult'] ,
                         'target': ['target_vocab_overlap', 'target_Relevance', 'target_Coherence', 'target_Consistency',
                                    'target_Fluency', 'target_bert_precision', 'target_bert_recall', 'target_bert_f1']
                         }
def xgboost(X_train, X_test, y_train, y_test ):

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # Define hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

    n = 10
    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
        evals=evals,
        verbose_eval=1
    )

    preds = model.predict(dtest_reg)
    rmse = root_mean_squared_error(y_test, preds)
    # print(f"RMSE of the base model: {rmse:.3f}")

    # Step 5: Evaluate the model's accuracy on the test set
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    #print(f"Mean Squared Error: {mse:.2f}")
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print(f"R^2 Score: {r2:.2f}")
    return {'xgboost-mse': float(round(mse,2)), 'xgboost-mae': float(round(mae,2)),
            "xgboost-rmse": float(round(rmse,2)), "xgboost-r2":float(round(r2,2))}


def ridge_regression(X_train, X_test, y_train, y_test ):


    # Instantiate the Ridge Regression model
    ridge_reg = Ridge(alpha=0.5)  # You can change the alpha parameter to add more or less regularization

    # Train the model
    ridge_reg.fit(X_train, y_train, max_iter= epochs)

    # Make predictions
    y_pred = ridge_reg.predict(X_test)

    # Evaluate the model
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    #print("Mean Squared Error (MSE):", mse)
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print("R² Score:", r2)

    # Optional: Display the coefficients
    #print("Coefficients:", ridge_reg.coef_)
    #print("Intercept:", ridge_reg.intercept_)
    scores = {'ridge-mse': float(round(mse,2)), 'ridge-mae': float(round(mae,2)),
            "ridge-rmse": float(round(rmse,2)), "ridge-r2": float(round(r2,2))}
    print(scores)
    return scores
def lasso_regression(X_train, X_test, y_train, y_test ):

    # Instantiate the Ridge Regression model
    lasso_reg = linear_model.Lasso(alpha=0.05)  # You can change the alpha parameter to add more or less regularization

    # Train the model
    lasso_reg.fit(X_train, y_train, max_iter= epochs)

    # Make predictions
    y_pred = lasso_reg.predict(X_test)

    # Evaluate the model
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    #print("Mean Squared Error (MSE):", mse)
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print("R² Score:", r2)

    # Optional: Display the coefficients
    #print("Coefficients:", lasso_reg.coef_)
    #print("Intercept:", lasso_reg.intercept_)
    scores = {'lasso-mse': float(round(mse,2)), 'lasso-mae': float(round(mae,2)),
            "lasso-rmse": float(round(rmse,2)), "lasso-r2": float(round(r2,2))}
    print (scores)
    return scores

def linear_regression(X_train, X_test, y_train, y_test ):

    # Instantiate the Ridge Regression model
    reg = LinearRegression() # You can change the alpha parameter to add more or less regularization

    # Train the model
    reg.fit(X_train, y_train, max_iter= epochs)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate the model
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    #print("Mean Squared Error (MSE):", mse)
    #print(f"Mean Absolute Error: {mae:.2f}")
    #print("R² Score:", r2)

    # Optional: Display the coefficients
    #print("Coefficients:", reg.coef_)
    #print("Intercept:", reg.intercept_)
    scores = {'mse': float(round(mse,2)), 'mae': float(round(mae,2)), "rmse": float(round(rmse,2)), "r2": float(round(r2,2))}
    print (scores)
    return scores


def weighted_average(nums, weights):
  return sum(x * y for x, y in zip(nums, weights)) / sum(weights)
def weighted_average_list(nums, weights):
  results = []
  for num in nums:
      res = sum(x * y for x, y in zip(num, weights)) / sum(weights)
      results.append(res)
  return results


def derive_baseline_features(df):
    df = df[domain_specific_features + baseline_feature_source + baseline_feature_target]

    feature_weight = [1 / len(baseline_feature_target)] * len(baseline_feature_target)

    weighted_y_target = weighted_average_list((df[baseline_feature_target]).values, feature_weight)
    weighted_y_source = weighted_average_list(df[baseline_feature_source].values, feature_weight)

    y_drop = np.subtract(weighted_y_source, weighted_y_target)

    df['weighted_y_target'] = weighted_y_target
    df['weighted_y_source'] = weighted_y_source
    df['y_drop'] = y_drop

    return df
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def normalize_features(df):
    def norm(df, features_to_normalize, update_y = None, a = None):
        for feature in features_to_normalize:
            if feature in df.columns:
                numbers = np.array(df[feature]).reshape((-1,1))
                df[feature] = NormalizeData(numbers)
        weighted_y_col = []
        if a is not None:
            for col in df.columns:
                if a in col:
                    weighted_y_col.append(col)

            # update weighted y
            feature_weight = [1 / len(weighted_y_col)] * len(weighted_y_col)

            df[update_y] = weighted_average_list((df[weighted_y_col]).values, feature_weight)

        return df

    #print (df.columns)
    # normalize all
    df = norm(df, features_to_normalize['all'])
    # normalize target
    df = norm(df, features_to_normalize['target'], update_y= 'y_weighted_target', a = 'target_')
    # normalize source
    df = norm(df, features_to_normalize['source'], update_y='y_weighted_source', a= 'source_')
    df['y_drop'] = df['y_weighted_source'] - df['y_weighted_target']
    return df

def run_regression(df:pd.DataFrame, mode:str):
    if mode == "baseline-raw" or mode == 'baseline-norm':
        print(mode)
        features_to_drop = baseline_feature_target + ['weighted_y_target']
    elif mode == 'all-raw' or mode == 'all-norm':
        print (mode)
        features_to_drop = ['y_weighted_target', 'target_bert_f1',  'target_rouge1', 'target_rouge2',
                            'target_rougeL', 'target_vocab_overlap','target_Relevance', 'target_Coherence',
                            'target_Consistency', 'target_Fluency','da-type','source', 'target',
                            'target_fs_grounded', 'Unnamed: 0',
                            'target_bert_precision', 'target_bert_recall',
                  ]
    else:
        print ("mode unknown. No Regression took place.")
        return

    df = df.dropna()
    df = df.drop(features_to_drop, axis=1)
    df = df.sample(frac=1)
    # Extract feature and target arrays
    X, y = df.drop('y_drop', axis=1), df[['y_drop']]
    # Extract text features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        X[col] = X[col].astype('category')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print ("Predictions with XGBoost")
    xgboost_scores =  xgboost(X_train, X_test, y_train, y_test)
    #xgboost_scores = {'xgboost-mse': 0, 'xgboost-mae': 0, "xgboost-rmse": 0, "xgboost-r2":0}

    print("Predictions with Linear Regression")
    reg_scores = linear_regression(X_train, X_test, y_train, y_test)
    print ("Predictions with Ridge Regression")
    ridge_scores = ridge_regression(X_train, X_test, y_train, y_test)

    print ("Predictions with Lasso Regression")
    lasso_scores = lasso_regression(X_train, X_test, y_train, y_test)

    ridge_scores.update(reg_scores)
    ridge_scores.update(xgboost_scores)
    ridge_scores.update(lasso_scores)
    feature_score = {'features':mode}
    feature_score.update(ridge_scores)

    return feature_score
def clear_cache():
    gc.collect()
    objects = [i for i in gc.get_objects()
               if isinstance(i, functools._lru_cache_wrapper)]

    # All objects cleared
    for object in objects:
        object.cache_clear()
if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--da_type',
                        type=str,
                        default="in-domain-adapt")
    parser.add_argument('--domain',
                        dest="domains",
                        action='append',
                        default=['arxiv', 'pubmed', 'govreport', 'wispermed', 'cnndm', 'samsum', 'bigpatent',
                                 'billsum', ])

    parser.add_argument('--template_path',
                        type=str,
                        default="overall_summary_ds_14_llama3.1_8b_zeroshot.xlsx")

    args = parser.parse_args()
    num_samples = 100
    experiment = '0-shot'
    total_domains = 13
    minumum_domains = 6
    cache = True
    epochs = 20

    all_scores = None
    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    directory = f"training_features/{date_time}"
    cache_directory = "training_features/2024-11-21_15-40-46"
    if cache:
        directory = cache_directory


    for n in range(minumum_domains,total_domains+1,1):
        print(f"Number of domains: {n}")
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = f"training_features_ds_{n}_llama3.1_8b_{experiment}_samples{num_samples}.xlsx"
        file_name = os.path.join(directory,file_name)
        if cache and os.path.isfile(file_name):
            features = pd.read_excel(file_name)
        else:
            features = construct_training_corpus(num_domains = n, da_type=args.da_type,
                                                template_path=args.template_path, num_samples=num_samples)

            features.to_excel(file_name)

        # 1) Prepare Baseline Features

        # 1.1) Raw features
        features_baseline = derive_baseline_features(features)
        #print (f"Baseline Features: {features_baseline.columns}")
        scores_baseline_raw = run_regression(features_baseline, mode='baseline-raw')

        # 1.2) Normalized features -> check if even needed
        features_baseline_norm = normalize_features(features_baseline)
        scores_baseline_norm = run_regression(features_baseline_norm, mode='baseline-norm')

        # 2) Prepare normal features

        # 2.1) Raw Features
        features = pd.read_excel(file_name)
        #print(f"All Features: {features.columns}")
        scores_all_raw = run_regression(features, mode='all-raw')

        # 2.2) Normalized Features
        features_norm = normalize_features(features)
        scores_all_norm = run_regression(features_norm, mode='all-norm')

        pd_scores = pd.DataFrame.from_records([scores_baseline_raw, scores_baseline_norm, scores_all_raw, scores_all_norm])
        pd_scores['num_datasets'] = [n] * len(pd_scores)
        #print (pd_scores)
        file_name = f"scores_ds_{n}_llama3.1_8b_{experiment}_{num_samples}.xlsx"
        file_name = os.path.join(directory, file_name)
        pd_scores.to_excel(file_name)
        if all_scores is None:
            all_scores = pd_scores
            #print (pd_scores.columns)
        else:
            all_scores = pd.concat([all_scores, pd_scores], axis=0)

    file_name = f"scores_llama3.1_8b_{experiment}_{num_samples}.xlsx"
    file_name = os.path.join(directory, file_name)
    #all_scores = all_scores[['num_datasets', 'features', 'ridge-mse', 'ridge-mae', "ridge-rmse", 'ridge-r2',
    #                         'lasso-mse', 'lasso-mae', "lasso-rmse", 'lasso-r2', 'xgboost-mse', 'xgboost-mae',
    #                         "xgboost-rmse", 'xgboost-r2',  ]]
    all_scores.to_excel(file_name)
    print (f"final scores stored at: {file_name}")
clear_cache()









