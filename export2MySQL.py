from mysql.connector import MySQLConnection, Error  # establish connection to mysql
from mysql.connector import MySQLConnection  # using Class for function call
import pandas as pd
from config import read_config  # calling config files that stored user credentials
import csv

# Change name of functions from save_data to create_empty_table or the like

# Display df settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "center")
pd.set_option("display.precision", 3)

# CSV files path
train_data_path = r"C:\Users\Doppler\Kaggle_Datasets\rossmann-store-sales\train.csv"
test_data_path = r"C:\Users\Doppler\Kaggle_Datasets\rossmann-store-sales\test.csv"
store_data_path = r"C:\Users\Doppler\Kaggle_Datasets\rossmann-store-sales\store.csv"
sample_submission_data_path = (
    r"C:\Users\Doppler\Kaggle_Datasets\rossmann-store-sales\sample_submission.csv"
)

file_paths_dict = {
    "train": train_data_path,
    "store": store_data_path,
    "test": test_data_path,
    "sample_submission": sample_submission_data_path,
}


def save_train_data(cursor):
    # show sample of train_data
    train_df = pd.read_csv(train_data_path, dtype=str)
    # print(train_df.head(100))
    train_df_cols = train_df.columns.tolist()
    # print(train_df_cols, len(train_df_cols))

    # Create table for train_data
    cursor.execute(
        f"""CREATE TABLE train(
    {train_df_cols[0]} VARCHAR(255),
    {train_df_cols[1]} VARCHAR(255),
    {train_df_cols[2]} DATE,
    {train_df_cols[3]} DOUBLE,
    {train_df_cols[4]} INT,
    {train_df_cols[5]} BINARY,
    {train_df_cols[6]} BINARY,
    {train_df_cols[7]} VARCHAR(255),
    {train_df_cols[8]} VARCHAR(255)
    )
    """
    )


def save_test_data(cursor):
    # show sample of test_data
    test_df = pd.read_csv(test_data_path, dtype=str)
    # print(test_df.head(100))
    test_df_cols = test_df.columns.tolist()
    # print(test_df_cols, len(test_df_cols))

    # Create table for test_data
    cursor.execute(
        f"""CREATE TABLE test(
    {test_df_cols[0]} VARCHAR(255),
    {test_df_cols[1]} VARCHAR(255),
    {test_df_cols[2]} VARCHAR(255),
    {test_df_cols[3]} DATE,
    {test_df_cols[4]} BINARY,
    {test_df_cols[5]} BINARY,
    {test_df_cols[6]} VARCHAR(255),
    {test_df_cols[7]} VARCHAR(255)
    )
    """
    )


def save_store_data(cursor):
    # show sample of store_data
    store_df = pd.read_csv(store_data_path, dtype=str)
    # print(store_df.head(100))
    store_df_cols = store_df.columns.tolist()
    # print(store_df_cols, len(store_df_cols))

    # Create table for test_data
    cursor.execute(
        f"""CREATE TABLE store(
    {store_df_cols[0]} VARCHAR(255),
    {store_df_cols[1]} VARCHAR(255),
    {store_df_cols[2]} VARCHAR(255),
    {store_df_cols[3]} DOUBLE,
    {store_df_cols[4]} VARCHAR(255),
    {store_df_cols[5]} VARCHAR(255),
    {store_df_cols[6]} BINARY,
    {store_df_cols[7]} VARCHAR(255),
    {store_df_cols[8]} VARCHAR(255),
    {store_df_cols[9]} VARCHAR(255)
    )
    """
    )


def save_sample_submission_data(cursor):
    # show sample of store_data
    sample_submission_df = pd.read_csv(sample_submission_data_path, dtype=str)
    # print(store_df.head(100))
    sample_submission_df_cols = sample_submission_df.columns.tolist()
    # print(sample_submission_df_cols, len(sample_submission_df_cols))

    # Create table for test_data
    cursor.execute(
        f"""CREATE TABLE sample_submission(
    {sample_submission_df_cols[0]} VARCHAR(255),
    {sample_submission_df_cols[1]} DOUBLE
    )
    """
    )


def query_with_fetchone(config):
    # Initialize variables for cursor and connection
    cursor = None
    conn = None

    try:
        # Establish a connection to the MySQL database using the provided configuration
        conn = MySQLConnection(**config)

        # Create a cursor to interact with the database
        cursor = conn.cursor()

        # ############### Create empty tables with desired names and datatypes ###############
        # save_train_data(cursor)
        # save_test_data(cursor)
        # save_store_data(cursor)
        # save_sample_submission_data(cursor)

        # Ensure we stored all tables
        cursor.execute("SHOW TABLES")
        [print(table) for table in cursor.fetchall()]

        ############### SQL import settings ###############
        # Turn this on, if you have issues sending data
        cursor.execute("SET GLOBAL local_infile=1")
        # This checks if you have issues with exporting later due to security:
        cursor.execute('SHOW VARIABLES LIKE "secure_file_priv"')
        [print(output) for output in cursor.fetchall()]
        # Want this ON, so we can transfer data depending on how you setup Import to MySQL
        cursor.execute("SHOW GLOBAL VARIABLES LIKE 'local_infile'")
        [print(output) for output in cursor.fetchall()]

        ############### store CSV files in MySQL tables ###############
        for file_name, file_path in file_paths_dict.items():
            with open(file_path) as data_raw:
                reader = csv.reader(data_raw, delimiter=",")
                data_list = list(reader)
            data_columns = data_list[0]

            # Clean up data rows: empty strings throw errors, replace empty strings with None,
            data_rows = [
                [None if value == "" else value for value in row]
                for row in data_list[1:]
            ]
            # Insert the data
            insert_query = f"""INSERT INTO {file_name} ({', '.join(data_columns)})
                                VALUES ({', '.join(['%s'] * len(data_columns))})"""
            cursor.executemany(
                insert_query, data_rows
            )  # first list is just column names, we don't want that!

            ## Sanity Check
            # cursor.execute("SELECT * from train LIMIT 10")
            # [print(output) for output  in cursor.fetchall()]

            ## Commit the changes made in MySQL
            conn.commit()

    except Error as e:
        # Print an error message if an error occurs during the execution of the query
        print(e)

    finally:
        # Close the cursor and connection in the 'finally' block to ensure it happens
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    # Read the database configuration from the 'config' module
    config = read_config()

    # Call the function with the obtained configuration to execute the query
    query_with_fetchone(config)
