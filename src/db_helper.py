# db_helper.py
# PURPOSE: One reusable function to run any SQL query
# and return a pandas DataFrame. Use this in every script.

import pandas as pd
import sqlite3

DB_PATH = "data/db/olist.db"

def run_query(sql: str) -> pd.DataFrame:
    """Run a SQL query and return result as a DataFrame."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)

def save_query(sql: str, output_path: str):
    """Run a SQL query and save result as CSV."""
    df = run_query(sql)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows → {output_path}")
    return df