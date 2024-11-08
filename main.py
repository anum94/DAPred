import argparse
import numpy as np
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from features.features import construct_training_corpus

baseline_feature_target = ['target_rouge1', 'target_rouge2', 'target_rougeL',
                           'target_vocab_overlap']
baseline_feature_source = ['source_rouge1', 'source_rouge2', 'source_rougeL',
                           'source_vocab_overlap']
domain_specific_features = ['learning_difficult', 'word-overlap', 'vocab-overlap', 'relevance-overlap',
                            'renyi-divergence', 'kl-divergence', 'js-divergence', ]
features_to_normalize = {'source': ['source_bert_precision', 'source_bert_recall', 'source_bert_f1', 'source_vocab_overlap',
                            'source_Relevance', 'source_Coherence', 'source_Consistency', 'source_Fluency'],
                         'all': ['renyi-divergence', 'kl-divergence', 'js-divergence',] ,
                         'target': ['target_vocab_overlap', 'target_Relevance', 'target_Coherence', 'target_Consistency',
                                    'target_Fluency', 'target_bert_precision', 'target_bert_recall', 'target_bert_f1']
                         }
def xgboost(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # Define hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

    n = 100
    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
        evals=evals,
        verbose_eval=10
    )

    preds = model.predict(dtest_reg)
    rmse = mean_squared_error(y_test, preds, squared=True)
    # print(f"RMSE of the base model: {rmse:.3f}")

    # Step 5: Evaluate the model's accuracy on the test set
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R^2 Score: {r2:.2f}")


def linear_regression(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the Ridge Regression model
    ridge_reg = Ridge(alpha=1.0)  # You can change the alpha parameter to add more or less regularization

    # Train the model
    ridge_reg.fit(X_train, y_train)

    # Make predictions
    y_pred = ridge_reg.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    # Optional: Display the coefficients
    print("Coefficients:", ridge_reg.coef_)
    print("Intercept:", ridge_reg.intercept_)

def weighted_average(nums, weights):
  return sum(x * y for x, y in zip(nums, weights)) / sum(weights)
def weighted_average_list(nums, weights):
  results = []
  for num in nums:
      res = sum(x * y for x, y in zip(num, weights)) / sum(weights)
      results.append(res)
  return results


def derive_baseline_features(feature_path):
    df = pd.read_excel(feature_path)
    df = df[domain_specific_features + baseline_feature_source + baseline_feature_target]

    feature_weight = [1 / len(baseline_feature_target)] * len(baseline_feature_target)

    weighted_y_target = weighted_average_list((df[baseline_feature_target]).values, feature_weight)
    weighted_y_source = weighted_average_list(df[baseline_feature_source].values, feature_weight)

    y_drop = np.subtract(weighted_y_source, weighted_y_target)

    df['weighted_y_target'] = weighted_y_target
    df['weighted_y_source'] = weighted_y_source
    df['y_drop'] = y_drop

    df = df.drop(baseline_feature_target + ['weighted_y_target'], axis=1)

    df = df.sample(frac=1)

    # Extract feature and target arrays
    X, y = df.drop('y_drop', axis=1), df[['y_drop']]
    return X, y
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def normalize_features(df):
        def norm(df, features_to_normalize, update_y = None, a = None):
            for feature in features_to_normalize:
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
    #diamonds = construct_training_corpus(domains=args.domains, da_type=args.da_type,
    #                                    template_path=args.template_path)
    #print(diamonds.describe())
    file_name = "training_features_ds_14_llama3.1_8b_zeroshot_100.xlsx"
    #diamonds.to_excel(file_name)
    diamonds = pd.read_excel(file_name)
    diamonds_norm = normalize_features(diamonds)

    diamonds = diamonds.drop(['y_weighted_target', 'target_bert_f1',
                              'target_rouge1', 'target_rouge2', 'target_rougeL',
                              'target_vocab_overlap',
                              'target_Relevance',
                              'target_Coherence',
                              'target_Consistency',
                              'target_Fluency',
                              'da-type',
                              'source',
                              'target', 'learning_difficult',
                            'target_fs_grounded', 'Unnamed: 0', 'learning_difficult',
                              'target_bert_precision', 'target_bert_recall',
                              ], axis=1)

    diamonds = diamonds.sample(frac=1)

    # Extract feature and target arrays
    X, y = diamonds.drop('y_drop', axis=1), diamonds[['y_drop']]
    # Extract text features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        X[col] = X[col].astype('category')
    # print (X.dtypes)

    X_base, y_base = derive_baseline_features(file_name)
    print ("Baseline xgboost")
    xgboost(X_base,y_base)
    print(" xgboost")

    xgboost(X, y)
    print("Baseline Regression model with only N-gram based features")
    linear_regression(X_base, y_base)
    print("Regression with model only all features")
    linear_regression(X, y)

    print ("With Features normalized")
    diamonds_norm = diamonds_norm.drop(['y_weighted_target', 'target_bert_f1',
                              'target_rouge1', 'target_rouge2', 'target_rougeL',
                              'target_vocab_overlap',
                              'target_Relevance',
                              'target_Coherence',
                              'target_Consistency',
                              'target_Fluency',
                              'da-type',
                              'source',
                              'target', 'learning_difficult',
                              'target_fs_grounded', 'Unnamed: 0',
                              'target_bert_precision', 'target_bert_recall',
                              ], axis=1)

    diamonds_norm = diamonds_norm.sample(frac=1)

    # Extract feature and target arrays
    X, y = diamonds_norm.drop('y_drop', axis=1), diamonds_norm[['y_drop']]
    # Extract text features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        X[col] = X[col].astype('category')
    # print (X.dtypes)

    X_base, y_base = derive_baseline_features(file_name)
    print ("Baseline xgboost")
    xgboost(X_base,y_base)
    print(" xgboost")

    xgboost(X, y)
    print("Baseline Regression model with only N-gram based features")
    linear_regression(X_base, y_base)
    print("Regression with model only all features")
    linear_regression(X, y)







