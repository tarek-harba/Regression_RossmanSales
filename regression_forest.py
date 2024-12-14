import numpy as np
import sklearn
import sklearn.model_selection

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import joblib
import os


def one_hot_encode(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    """
    Performs one-hot encoding on specified categorical columns in the DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the data to be encoded.
        categorical_columns (list[str]): A list of column names (strings) in the DataFrame to apply one-hot encoding to.

    Returns:
        pd.DataFrame: A new DataFrame with the categorical columns replaced by their one-hot encoded counterparts.
    """
    one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    for col_name in categorical_columns:
        encoded_col_df = one_hot_encoder.fit_transform(df[col_name].to_frame())
        df = pd.concat([df, encoded_col_df], axis=1).drop(columns=[col_name])
    return df


def preprocess(df: pd.DataFrame):
    """Implementing the RegressionForest, we first split into training and validation sets.
    Since we have alot of data (842206 samples) we limit the size of the validation set to
    just 5% of the available samples"""
    input_col_names = [
        "CompetitionDistance",
        "Assortment",
        "StoreType",
        "Promo",
        "MonthOfYear",
        "DayOfWeek",
        "Store_compressed",
    ]

    categorical_col_names = [
        "Assortment",
        "StoreType",
        "Promo",
        "MonthOfYear",
        "DayOfWeek",
        "Store_compressed",
    ]
    df_train, df_valid = sklearn.model_selection.train_test_split(
        df, train_size=0.95, random_state=1
    )

    x_train = df_train[input_col_names]
    y_train = df_train["Sales"]

    x_valid = df_valid[input_col_names]
    y_valid = df_valid["Sales"]

    x_train = one_hot_encode(x_train, categorical_col_names)
    x_valid = one_hot_encode(x_valid, categorical_col_names)

    return x_train, y_train, x_valid, y_valid


def rmspe_loss_fn(prediction, target):
    mask = target != 0
    prediction = prediction[mask]
    target = target[mask]
    loss = np.sqrt(np.mean(((target - prediction) / target) ** 2))
    return loss


def mse_loss_fn(prediction, target):
    mask = target != 0
    prediction = prediction[mask]
    target = target[mask]
    loss = np.mean((target - prediction) ** 2)
    return loss


def forest_init(x_train, y_train, x_valid=None, y_valid=None):
    x_train = x_train.values
    y_train = y_train.values

    if x_valid is not None:
        x_valid = x_valid.values
        y_valid = y_valid.values
        rng = np.random.RandomState(1)

    # for d in range(50):
    # for n in range(6):
    # n_estimators = (n+1)*5
    # depth = d+8
    if x_valid is not None:
        regression_forest = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=22),
            n_estimators=10,
            random_state=rng,  # depth ref  22, n_estimators ref 10
        )
        regression_forest.fit(x_train, y_train)
        valid_pred = regression_forest.predict(x_valid)
        MSE_loss_valid = mse_loss_fn(valid_pred, y_valid)
        RMSPE_loss_valid = rmspe_loss_fn(valid_pred, y_valid)
        print(
            f"Depth:{22:.4f}    -  n_estimators:{10:.4f}      - MSE Valid Loss:{MSE_loss_valid:.4f}"
            f"      RMSPE Valid Loss:{RMSPE_loss_valid:.4f}"
        )
    else:
        regression_forest = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=22),
            n_estimators=10,  # depth ref  22, n_estimators ref 10
        )
        regression_forest.fit(x_train, y_train)

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # model_path = f'{dir_path}/regression_forest.joblib'
        joblib.dump(regression_forest, "./regression_forest.joblib", compress=3)
        print("Forest Saved!")
