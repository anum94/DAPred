import argparse
import numpy as np
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import xgboost as xgb
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from features.features import construct_training_corpus

def xgboost(X, y ):
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

def linear_regression(X, y ):
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

if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--da_type',
                        type=str,
                        default="in-domain-adapt")
    parser.add_argument('--domain',
                        dest="domains",
                        action='append',
                        default=['arxiv', 'pubmed', 'govreport', 'wispermed', 'cnndm', 'samsum', 'bigpatent', 'billsum',])

    parser.add_argument('--template_path',
                        type=str,
                        default="overall_summary.xlsx")

    args = parser.parse_args()
    diamonds = construct_training_corpus(domains=args.domains, da_type=args.da_type,
                              template_path=args.template_path)
    print (diamonds.describe())
    diamonds.to_excel("training_features.xlsx")
    diamonds = pd.read_excel("training_features.xlsx")

    diamonds.drop('y_weighted_target',axis=1)
    diamonds.drop('target', axis=1)
    diamonds.drop('source', axis=1)
    diamonds.drop('learning_difficult', axis=1)
    print(diamonds.describe())
    diamonds = diamonds.sample(frac = 1)

    # Extract feature and target arrays
    X, y = diamonds.drop('y_drop', axis=1), diamonds[['y_drop']]
    # Extract text features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        X[col] = X[col].astype('category')
    #print (X.dtypes)

    xgboost(X,y)
    linear_regression(X,y)



