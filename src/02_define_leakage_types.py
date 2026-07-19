# 02_define_leakage_types.py
# PURPOSE: Document what each leakage type means in terms of THIS dataset's columns.
# This file is your "business requirements" document — write it carefully.

LEAKAGE_DEFINITIONS = {

    "L1_discount_abuse": {
        "description": "Orders where the payment value is significantly lower than "
                       "the sum of item prices, indicating unusual discounting.",
        "formula":     "discount_depth = (sum(price) - payment_value) / sum(price)",
        "threshold":   "Flag if discount_depth > 0.30 (more than 30% off total price)",
        "tables":      ["order_items", "order_payments"],
        "join_key":    "order_id",
        # WHY THIS MATTERS: A legitimate discount is ~5-15%. 30%+ usually means
        # a coupon was misapplied, a pricing error, or internal fraud.
    },

    "L2_refund_spike": {
        "description": "Product categories where refund/cancellation rate spikes "
                       "above their own historical baseline in a given week.",
        "formula":     "refund_rate = cancelled_orders / total_orders per category per week",
        "threshold":   "Flag if refund_rate > (rolling_8wk_avg + 2 * rolling_8wk_std)",
        "tables":      ["orders", "order_items", "products", "category_xlat"],
        "join_key":    "order_id → product_id → product_category_name",
        # WHY THIS MATTERS: A spike in one category (not overall) often means
        # a bad batch of products, a supplier issue, or a listing error.
    },

    "L3_silent_churn": {
        "description": "High-value customers (top 20% by lifetime spend) who placed "
                       "their last order 6+ months ago and have not returned.",
        "formula":     "days_since_last_order = today - max(order_purchase_timestamp)",
        "threshold":   "Flag if customer in top_20pct_LTV AND days_since_last_order > 180",
        "tables":      ["orders", "customers", "order_payments"],
        "join_key":    "customer_id",
        # WHY THIS MATTERS: Losing a top-20% customer = losing ~80% of their
        # revenue contribution. Companies often don't notice until it's too late.
    },

    "L4_short_lifecycle_orders": {
        "description": "Orders that were delivered but returned/refunded within 30 days, "
                       "concentrated in specific seller-category combinations.",
        "formula":     "lifecycle_days = order_delivered_customer_date - order_purchase_timestamp",
        "threshold":   "Flag seller-category pairs where >25% of orders have lifecycle < 30 days",
        "tables":      ["orders", "order_items", "products", "sellers"],
        "join_key":    "order_id → seller_id + product_category_name",
        # WHY THIS MATTERS: Short lifecycle = the product didn't stick. These
        # seller-category combos are silently burning acquisition cost.
    }
}

# Print a summary to confirm your definitions
for key, val in LEAKAGE_DEFINITIONS.items():
    print(f"\n{key}")
    print(f"  What:      {val['description'][:80]}...")
    print(f"  Measure:   {val['formula']}")
    print(f"  Flag when: {val['threshold']}")