# 01_explore_data.py
# PURPOSE: Understand every table before touching the data
# Run this FIRST. Never skip this step.

import pandas as pd
import os

DATA_PATH = r"C:\Users\suyas\OneDrive\Documents\Desktop\Revenue-leakage-detector\data\raw"

# ── Load all 9 tables ──────────────────────────────────────────
tables = {
    "orders":        "olist_orders_dataset.csv",
    "customers":     "olist_customers_dataset.csv",
    "order_items":   "olist_order_items_dataset.csv",
    "order_payments":"olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products":      "olist_products_dataset.csv",
    "sellers":       "olist_sellers_dataset.csv",
    "geolocation":   "olist_geolocation_dataset.csv",
    "category_xlat": "product_category_name_translation.csv",
}

dfs = {}
for name, file in tables.items():
    df = pd.read_csv(os.path.join(DATA_PATH, file))
    dfs[name] = df
    print(f"\n{'='*50}")
    print(f"TABLE: {name}")
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Sample:\n{df.head(2)}")