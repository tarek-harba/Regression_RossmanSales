from mysql.connector import MySQLConnection, Error  # establish connection to mysql
from mysql.connector import MySQLConnection  # using Class for function call
import pandas as pd
from config import read_config  # calling config files that stored user credentials
import csv

# Problems (1) does not save column names (2) uses space not comma or separation


def import_from_mysql(config, tables_names):
    """
    Exports data from specified MySQL tables into CSV files.
    """
    # Initialize variables for cursor and connection
    cursor = None
    conn = None

    try:
        # Establish a connection to the MySQL database using the provided configuration
        conn = MySQLConnection(**config)

        # Create a cursor to interact with the database
        cursor = conn.cursor()

        # Ensure we have all tables
        # cursor.execute("SHOW TABLES")
        # [print(table) for table in cursor.fetchall()]
        for table_name in tables_names:
            outfile = f"""
            SELECT * FROM {table_name} INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/rossman_data/{table_name}.csv' 
            FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\\n';
            """
            cursor.execute(outfile)

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

    tables_names = ["train", "test", "store", "sample_submission"]
    # Call the function with the obtained configuration to execute the query
    import_from_mysql(config, tables_names)
