import argparse
import numpy as np
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
#import xgboost as xgb
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

    print(df.describe())
    df = df.sample(frac=1)

    # Extract feature and target arrays
    X, y = df.drop('y_drop', axis=1), df[['y_drop']]
    return X, y


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
    diamonds = construct_training_corpus(domains=args.domains, da_type=args.da_type,
                                        template_path=args.template_path)
    print(diamonds.describe())
    file_name = "training_features_llama_3.1_ds_9.xlsx"
    #diamonds.to_excel(file_name)
    diamonds = pd.read_excel(file_name)

    diamonds = diamonds.drop(['y_weighted_target', 'target_bert_f1',
                              'target_rouge1', 'target_rouge2', 'target_rougeL',
                              'target_vocab_overlap',
                              'target_Relevance',
                              'target_Coherence',
                              'target_Consistency',
                              'target_Fluency',
                              'da-type',
                              'source',
                              'target',
                            'target_fs_grounded', 'Unnamed: 0',
                              'target_bert_precision', 'target_bert_recall',
                              ], axis=1)

    print(diamonds.describe())
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
    #print ("Baseline xgboost")
    #xgboost(X_base,y_base)
    #print(" xgboost")

    #xgboost(X, y)
    print("Baseline Regression")
    linear_regression(X_base, y_base)
    print("Regression")
    linear_regression(X, y)



