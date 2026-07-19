# 11_evaluate_detector.py
# PURPOSE: Measure how accurately our detectors caught the
# injected leakage events. This is your model evaluation section.

import pandas as pd
import numpy as np

# Load ground truth from Phase 1
ground_truth = pd.read_csv("data/processed/leakage_ground_truth.csv")
orders       = pd.read_csv("data/processed/orders_with_leakage.csv")
items        = pd.read_csv("data/raw/olist_order_items_dataset.csv")
products     = pd.read_csv("data/raw/olist_products_dataset.csv")

# Load our anomaly flags
zscore_flags = pd.read_csv("data/outputs/anomalies/zscore_anomalies.csv")
if_flags     = pd.read_csv(
    "data/outputs/anomalies/high_confidence_anomalies.csv"
)

# ── Map ground truth orders → category + week ─────────────────
# Ground truth is at order level. Our flags are at category-week level.
# We need to join them to compare.

gt_orders = ground_truth.merge(
    orders[["order_id","order_purchase_timestamp"]], on="order_id", how="left"
)
gt_orders["order_purchase_timestamp"] = pd.to_datetime(
    gt_orders["order_purchase_timestamp"]
)
gt_orders["year_week"] = (gt_orders["order_purchase_timestamp"]
                           .dt.strftime("%Y-W%W"))

# Add category info
gt_with_cat = gt_orders.merge(
    items[["order_id","product_id"]], on="order_id", how="left"
).merge(
    products[["product_id","product_category_name"]], on="product_id", how="left"
)

# Summarize: which category-weeks have injected leakage?
gt_category_weeks = (
    gt_with_cat
    .groupby(["year_week","product_category_name","leakage_type"])
    .size()
    .reset_index(name="injected_count")
)

print("=== GROUND TRUTH SUMMARY ===")
print(f"Total injected leakage events : {len(ground_truth)}")
print(f"Leakage types:\n{ground_truth['leakage_type'].value_counts().to_string()}")
print(f"\nCategory-weeks with injected leakage: {len(gt_category_weeks)}")


# ── Precision & Recall for Z-score detector ───────────────────
# Precision = of all flags raised, how many were real leakage?
# Recall    = of all real leakage events, how many did we catch?

def evaluate_detector(flags_df, gt_df, detector_name):
    # Flagged category-weeks
    flagged = set(zip(flags_df["year_week"], flags_df["category_en"]))

    # Ground truth category-weeks
    actual = set(zip(gt_df["year_week"], gt_df["product_category_name"]))

    true_positives  = flagged & actual   # caught real leakage
    false_positives = flagged - actual   # flagged but not real leakage
    false_negatives = actual - flagged   # missed real leakage

    precision = len(true_positives) / len(flagged) if flagged else 0
    recall    = len(true_positives) / len(actual)  if actual  else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print(f"\n=== {detector_name} EVALUATION ===")
    print(f"  Total flags raised  : {len(flagged)}")
    print(f"  True positives      : {len(true_positives)}")
    print(f"  False positives     : {len(false_positives)}")
    print(f"  False negatives     : {len(false_negatives)}")
    print(f"  Precision           : {precision:.1%}")
    print(f"  Recall              : {recall:.1%}")
    print(f"  F1 Score            : {f1:.1%}")

    return {
        "detector": detector_name,
        "flags_raised": len(flagged),
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }

zscore_eval = evaluate_detector(zscore_flags, gt_category_weeks, "Z-Score")
if_eval     = evaluate_detector(if_flags,     gt_category_weeks, "Isolation Forest")

# Save evaluation results
eval_df = pd.DataFrame([zscore_eval, if_eval])
eval_df.to_csv("data/outputs/anomalies/detector_evaluation.csv", index=False)

print("\n=== SIDE-BY-SIDE COMPARISON ===")
print(eval_df[["detector","precision","recall","f1_score"]].to_string(index=False))
print("\nSaved → data/outputs/anomalies/detector_evaluation.csv")