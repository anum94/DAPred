import argparse
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
#import xgboost as xgb
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
from features.features import construct_training_corpus

if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--da_type',
                        type=str,
                        default="in-domain-adapt")
    parser.add_argument('--domain',
                        dest="domains",
                        action='append',
                        default=['arxiv', 'pubmed', 'govreport', 'wispermed', 'cnndm', 'samsum', 'bigpatent'])

    parser.add_argument('--template_path',
                        type=str,
                        default="overall_summary.xlsx")

    args = parser.parse_args()
    diamonds = construct_training_corpus(domains=args.domains, da_type=args.da_type,
                              template_path=args.template_path)
    print (diamonds.describe())
    diamonds.to_excel("Diamonds.xlsx")
    diamonds.drop('y_weighted_target',axis=1)
    diamonds.drop('target', axis=1)
    diamonds.drop('source', axis=1)



    # Extract feature and target arrays
    X, y = diamonds.drop('y_drop', axis=1), diamonds[['y_drop']]
    # Extract text features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        X[col] = X[col].astype('category')
    print (X.dtypes)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    '''
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
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"RMSE of the base model: {rmse:.3f}")
    '''

