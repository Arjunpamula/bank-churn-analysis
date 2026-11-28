import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. LOAD -------------------------------------------------------------------
df = pd.read_csv("churn_clean.csv")
baseline = df["churn"].mean()
total_churners = df["churn"].sum()
MIN_N = 100  # ignore combinations with fewer customers: rates are unstable

ORDER = {
    "age_group": ["18-30", "31-40", "41-50", "51-60", "60+"],
    "product_group": ["1 product", "2 products", "3+ products"],
    "country": ["France", "Spain", "Germany"],
    "active_label": ["Active", "Inactive"],
    "gender": ["Female", "Male"],
    "balance_band": ["Zero", "Low", "Medium", "High"],
}

# 2. THE PAIRS WE WANT TO TEST ---------------------------------------------
PAIRS = [
    ("country", "product_group"),
    ("age_group", "active_label"),
    ("product_group", "active_label"),
    ("country", "age_group"),
    ("country", "active_label"),
    ("gender", "active_label"),
    ("age_group", "product_group"),
]


# 3. TWO-WAY TABLE: CHURN RATE AND SIZE ------------------------------------
def two_way(row_col, col_col):
    g = df.groupby([row_col, col_col])["churn"].agg(customers="count", churned="sum", rate="mean")
    rate = g["rate"].unstack(col_col).reindex(index=ORDER[row_col], columns=ORDER[col_col]) * 100
    n = g["customers"].unstack(col_col).reindex(index=ORDER[row_col], columns=ORDER[col_col])
    return rate, n


def heatmap(row_col, col_col):
    rate, n = two_way(row_col, col_col)
    masked = rate.where(n >= MIN_N)  # hide tiny cells
    fig, ax = plt.subplots(figsize=(1.6 * len(rate.columns) + 3, 0.7 * len(rate) + 2.5))
    im = ax.imshow(masked.values.astype(float), cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(rate.columns)), rate.columns)
    ax.set_yticks(range(len(rate.index)), rate.index)
    for i in range(len(rate.index)):
        for j in range(len(rate.columns)):
            r, c = rate.iloc[i, j], n.iloc[i, j]
            if pd.isna(c) or c < MIN_N:
                ax.text(j, i, f"n={0 if pd.isna(c) else int(c)}\n(too small)", ha="center", va="center", fontsize=7, color="grey")
            else:
                color = "white" if r > 55 else "black"
                ax.text(j, i, f"{r:.0f}%\nn={int(c):,}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title(f"Churn rate: {row_col} × {col_col}  (baseline {baseline:.1%})", fontsize=10)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    plt.colorbar(im, label="Churn rate (%)")
    plt.tight_layout()
    plt.savefig(f"heatmap_{row_col}_x_{col_col}.png", dpi=150)
    plt.close()


for a, b in PAIRS:
    rate, n = two_way(a, b)
    print(f"--- {a} x {b}: churn rate % ---")
    print(rate.round(1).to_string())
    print("customers:")
    print(n.to_string())
    print()
    heatmap(a, b)

# 4. RANK ALL COMBINATIONS (2-WAY AND 3-WAY) -------------------------------
rows = []
cols = list(ORDER)
for k in (2, 3):
    for combo in itertools.combinations(cols, k):
        g = df.groupby(list(combo))["churn"].agg(customers="count", churned="sum", rate="mean").reset_index()
        g = g[g["customers"] >= MIN_N]
        g["segment"] = g[list(combo)].astype(str).agg(" | ".join, axis=1)
        g["dimensions"] = " x ".join(combo)
        rows.append(g[["dimensions", "segment", "customers", "churned", "rate"]])

combos = pd.concat(rows, ignore_index=True)
combos["churn_rate_%"] = (combos["rate"] * 100).round(1)
combos["lift_vs_baseline"] = (combos["rate"] / baseline).round(2)
combos["share_of_all_churners_%"] = (combos["churned"] / total_churners * 100).round(1)
combos = combos.drop(columns="rate")
combos.to_csv("segment_combinations_all.csv", index=False)

print(f"--- Top 15 two-way segments by churn rate (n >= {MIN_N}) ---")
top2 = combos[combos["dimensions"].str.count(" x ") == 1].sort_values("churn_rate_%", ascending=False)
print(top2.head(15)[["segment", "customers", "churn_rate_%", "lift_vs_baseline", "share_of_all_churners_%"]].to_string(index=False))

# 5. PRIORITY LIST: HIGH RISK *AND* BIG ENOUGH TO MATTER -------------------
# Segments with at least 1.5x the baseline churn and 400+ customers
priority = top2[(top2["lift_vs_baseline"] >= 1.5) & (top2["customers"] >= 400)]
priority = priority.sort_values("share_of_all_churners_%", ascending=False)
priority.to_csv("priority_segments.csv", index=False)
print("\n--- Priority segments (lift >= 1.5, n >= 400), ordered by share of churners ---")
print(priority.head(12)[["segment", "customers", "churn_rate_%", "lift_vs_baseline", "share_of_all_churners_%"]].to_string(index=False))
