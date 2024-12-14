import pandas as pd
from pandas import DataFrame
from scipy.stats import alpha, f_oneway
from sqlalchemy import create_engine
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from matplotlib.animation import FuncAnimation

from eda import statistical_eda, visual_eda
import DNN
import regression_forest
import torch

from data_cleanup import data_cleanup


def pandas_display_settings() -> None:
    # Display df settings
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "center")
    pd.set_option("display.precision", 3)


def import_from_mysql_sqlalchemy(table_name: str):
    try:
        # The database URL must be in a specific format
        db_url = "mysql://{USER}:{PWD}@{HOST}/{DBNAME}"  # +mysqlconnector
        # Replace the values below with your own
        # DB username, password, host and database name
        db_url = db_url.format(
            USER="root", PWD="9895", HOST="localhost:3306", DBNAME="rossman_store_sales"
        )
        # Create the DB engine instance which connects to the database
        engine = create_engine(db_url, echo=False)

        return pd.read_sql_table(table_name, engine)

    except Exception as e:
        print(e)


def explore_df(df: pd.DataFrame) -> None:
    print(df.info())
    print(df.head())
    for column in df:
        print("\033[34m" + column + "\033[0m", df[column].unique())
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------"
        )


if __name__ == "__main__":
    pandas_display_settings()

    train_df = import_from_mysql_sqlalchemy("train")
    store_df = import_from_mysql_sqlalchemy("store")

    """We look at the number of rows and columns to see if they match what we expect"""
    explore_df(train_df)
    explore_df(store_df)

    """ Having seen the data, we add the columns of store_dataframe to train_dataframe by (store number column)"""
    merged_train_df = pd.merge(train_df, store_df, on="Store")
    explore_df(merged_train_df)  # Ensure merger is correct

    """First clean up the data, handling missing values and reformatting data to facilitate work later"""
    processed_train_df = data_cleanup(merged_train_df)

    """ Next, in order to know which features matter more for prediction, we conduct
    1- Visual assessment with bin, scatter and box plots 
    2- Statistical assessment with Pearson's correlation, ANalysis Of VAriance (ANOVA) 
    """
    processed_train_df = visual_eda(processed_train_df)

    """After the visual exploratory data analysis (EDA), we conduct a brief statistical EDA
    where we (a) look into a correlation heatmap between the numeric features and (b) implement ANOVA
    which tells us if Store feature is relevant for predicting the Sales.
       This is because we could not have visually explored the relation between Store and Sales given the great
       number of labels in Store with about 1115 labels."""
    processed_train_df = statistical_eda(processed_train_df)
    plt.show()
    """ We are now ready to implement a predictor, we start with a deep neural network"""
    x_train, y_train, x_valid, y_valid = DNN.preprocess(processed_train_df)
    DNN.dnn_init(x_train, y_train, x_valid, y_valid)

    complete_training_set_x = torch.cat([x_train, x_valid], 0)
    complete_training_set_y = torch.cat([y_train, y_valid], 0)
    DNN.dnn_init(complete_training_set_x, complete_training_set_y)

    """Adaregressor"""
    x_train, y_train, x_valid, y_valid = regression_forest.preprocess(
        processed_train_df
    )
    print(x_train.iloc[0, :])
    regression_forest.forest_init(x_train, y_train, x_valid, y_valid)

    complete_training_set_x = pd.concat([x_train, x_valid])
    complete_training_set_y = pd.concat([y_train, y_valid])
    regression_forest.forest_init(x_train, y_train)
