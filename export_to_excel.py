"""
export_to_excel.py

Usage:
    1. Make sure you have pull_requests.db in the same directory
       (or adjust the DB_NAME variable below).

    2. Install pandas and openpyxl if you haven't:
       pip install pandas openpyxl

    3. Run the script:
       python export_to_excel.py

    It will create (or overwrite) an Excel file named exported_data.xlsx
    with three sheets: PullRequests, PRFiles, and PRComments.
"""

import sqlite3
import pandas as pd

DB_NAME = "pull_requests.db"
OUTPUT_EXCEL = "exported_data.xlsx"

def main():
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_NAME)

    # List of tables you want to export
    tables = ["PullRequests", "PRFiles", "PRComments"]

    # Create an ExcelWriter object (uses openpyxl by default)
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        for table in tables:
            # Load the table into a pandas DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            # Write the DataFrame to an Excel sheet named after the table
            df.to_excel(writer, sheet_name=table, index=False)

    conn.close()
    print(f"Exported tables {tables} to {OUTPUT_EXCEL} successfully.")

if __name__ == "__main__":
    main()
