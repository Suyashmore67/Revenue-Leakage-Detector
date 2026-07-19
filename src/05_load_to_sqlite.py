# 05_load_to_sqlite.py
# PURPOSE: Load all processed CSVs into a SQLite database
# so we can write proper SQL queries against them.

import pandas as pd
import sqlite3
import os

os.makedirs("data/db", exist_ok=True)

# Connect — this creates the file if it doesn't exist
conn = sqlite3.connect("data/db/olist.db")

# ── Load all tables into SQLite ────────────────────────────────
tables = {
    "orders":           "data/processed/orders_with_leakage.csv",
    "order_payments":   "data/processed/payments_with_leakage.csv",
    "order_items":      "data/raw/olist_order_items_dataset.csv",
    "customers":        "data/raw/olist_customers_dataset.csv",
    "products":         "data/raw/olist_products_dataset.csv",
    "sellers":          "data/raw/olist_sellers_dataset.csv",
    "category_xlat":    "data/raw/product_category_name_translation.csv",
}

for table_name, filepath in tables.items():
    df = pd.read_csv(filepath)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Loaded {table_name:20s} → {len(df):>7,} rows")

conn.close()
print("\nDatabase saved to data/db/olist.db")