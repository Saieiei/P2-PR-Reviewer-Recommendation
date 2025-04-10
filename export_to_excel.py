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
    with three sheets: PullRequests, PRFiles, and PRComments, and show progress
    in the terminal.
"""

import sqlite3
import pandas as pd
import re

DB_NAME = "pull_requests.db"
OUTPUT_EXCEL = "exported_data.xlsx"

def clean_illegal_chars(x):
    """
    Remove characters that are illegal in Excel worksheets from string values.
    Excel does not allow control characters in a cell, so we strip those out.
    """
    if isinstance(x, str):
        # Remove control characters except tab, newline, and carriage return
        return re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1F]', '', x)
    return x

def main():
    print("Connecting to the database...")
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_NAME)
    
    # List of tables you want to export
    tables = ["PullRequests", "PRFiles", "PRComments"]
    
    print(f"Preparing to export tables: {tables}")
    
    # Create an ExcelWriter object (uses openpyxl by default)
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        for table in tables:
            print(f"Processing table '{table}'...")
            # Load the table into a pandas DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            print(f"Loaded '{table}' with {len(df)} rows.")
            
            # Clean data by removing illegal Excel characters
            df = df.applymap(clean_illegal_chars)
            
            # Write the DataFrame to an Excel sheet named after the table
            df.to_excel(writer, sheet_name=table, index=False)
            print(f"Exported table '{table}' to Excel sheet.")
    
    conn.close()
    print(f"All tables exported successfully to '{OUTPUT_EXCEL}'.")

if __name__ == "__main__":
    main()
