import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, confusion_matrix)

RANDOM = 42
df = pd.read_csv("churn_clean.csv")
baseline = df["churn"].mean()

# 1. FEATURES ----------------------------------------------------------------
# Products are NOT a straight line (2 is safest, 1 and 3+ are risky),
# so treat them as categories. Same idea for country.
X = pd.DataFrame({
    "age": df["age"],
    "credit_score": df["credit_score"],
    "tenure": df["tenure"],
    "balance": df["balance"],
    "estimated_salary": df["estimated_salary"],
    "active_member": df["active_member"],
    "is_female": (df["gender"] == "Female").astype(int),
    "has_credit_card": df["credit_card"],
    "zero_balance": (df["balance"] == 0).astype(int),
    "country_Germany": (df["country"] == "Germany").astype(int),
    "country_Spain": (df["country"] == "Spain").astype(int),
    "products_1": (df["products_number"] == 1).astype(int),
    "products_3plus": (df["products_number"] >= 3).astype(int),
})
y = df["churn"]

# 2. TRAIN / TEST SPLIT ------------------------------------------------------
# stratify keeps the ~20% churn share the same in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM)

# 3. TWO MODELS --------------------------------------------------------------
# Logistic regression: simple, explainable (gives odds ratios).
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
# Gradient boosting: can learn interactions (e.g. age x inactivity) automatically.
gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250,
                                     l2_regularization=1.0, random_state=RANDOM)

results = []
fitted = {}
for name, model in [("Logistic regression", logit), ("Gradient boosting", gbm)]:
    model.fit(X_train, y_train)
    fitted[name] = model
    p = model.predict_proba(X_test)[:, 1]
    results.append({"model": name,
                    "ROC_AUC": round(roc_auc_score(y_test, p), 3),
                    "PR_AUC": round(average_precision_score(y_test, p), 3)})
res = pd.DataFrame(results)
print("=== Model comparison on held-out test set (25% of customers) ===")
print(f"(random guessing: ROC_AUC 0.50, PR_AUC {y_test.mean():.2f})")
print(res.to_string(index=False))

best_name = res.sort_values("ROC_AUC", ascending=False).iloc[0]["model"]
best = fitted[best_name]
print(f"\nBest model: {best_name}")

# 4. IMBALANCE: ACCURACY IS MISLEADING --------------------------------------
# 80% of customers stay, so "predict everyone stays" is already 80% accurate.
print(f"\nAccuracy of 'predict nobody leaves': {(1 - y_test.mean()):.1%}  <- useless benchmark")

p_test = best.predict_proba(X_test)[:, 1]
rows = []
for thr in [0.5, 0.35, 0.25, 0.20]:
    pred = (p_test >= thr).astype(int)
    rows.append({"threshold": thr,
                 "precision_%": round(precision_score(y_test, pred) * 100, 1),
                 "recall_%": round(recall_score(y_test, pred) * 100, 1),
                 "customers_flagged_%": round(pred.mean() * 100, 1)})
print("\n=== Precision / recall at different alert thresholds (test set) ===")
print("precision = of those flagged, how many really leave; recall = of those who leave, how many we catch")
print(pd.DataFrame(rows).to_string(index=False))

# 5. SCORE EVERY CUSTOMER WITHOUT CHEATING ----------------------------------
# Out-of-fold: each customer is scored by a model that never saw that customer.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
df["churn_probability"] = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
print(f"\nOut-of-fold ROC_AUC on all customers: {roc_auc_score(y, df['churn_probability']):.3f}")

# 6. RISK TIERS --------------------------------------------------------------
df["risk_tier"] = pd.cut(df["churn_probability"], bins=[-0.01, 0.20, 0.50, 1.0],
                         labels=["Low", "Medium", "High"])
tiers = (df.groupby("risk_tier", observed=True)
           .agg(customers=("churn", "count"), actual_churners=("churn", "sum"),
                actual_churn_rate=("churn", "mean"),
                avg_predicted=("churn_probability", "mean"))
           .reset_index())
tiers["share_of_customers_%"] = (tiers["customers"] / len(df) * 100).round(1)
tiers["share_of_all_churners_%"] = (tiers["actual_churners"] / df["churn"].sum() * 100).round(1)
tiers["actual_churn_rate"] = (tiers["actual_churn_rate"] * 100).round(1)
tiers["avg_predicted"] = (tiers["avg_predicted"] * 100).round(1)
print("\n=== Risk tiers: do predicted tiers match reality? ===")
print(tiers.to_string(index=False))
tiers.to_csv("risk_tier_summary.csv", index=False)

# 7. WHAT TO MONITOR: FEATURE IMPORTANCE -------------------------------------
perm = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                              n_repeats=15, random_state=RANDOM)
imp = (pd.DataFrame({"feature": X.columns, "importance_auc_drop": perm.importances_mean,
                     "std": perm.importances_std})
         .sort_values("importance_auc_drop", ascending=False))
print("\n=== What drives risk: permutation importance (drop in AUC when feature is scrambled) ===")
print(imp.round(4).to_string(index=False))
imp.to_csv("feature_importance.csv", index=False)

# Logistic regression odds ratios (plain-English effects)
lr_model = fitted["Logistic regression"]
coefs = lr_model.named_steps["logisticregression"].coef_[0]
scale = lr_model.named_steps["standardscaler"].scale_
odds = pd.DataFrame({"feature": X.columns,
                     "odds_ratio_per_unit": np.exp(coefs / scale).round(3)})
print("\n=== Logistic regression odds ratios (per 1 unit; >1 raises risk, <1 lowers it) ===")
print(odds.to_string(index=False))

# 8. CHARTS ------------------------------------------------------------------
top = imp.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(top["feature"], top["importance_auc_drop"], xerr=top["std"], color="#c0392b")
ax.set_xlabel("Drop in ROC-AUC when feature is scrambled")
ax.set_title("What drives churn risk")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("chart_feature_importance.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4))
colors = {"Low": "#7f8c8d", "Medium": "#e67e22", "High": "#c0392b"}
ax.bar(tiers["risk_tier"].astype(str), tiers["actual_churn_rate"],
       color=[colors[t] for t in tiers["risk_tier"].astype(str)])
for i, r in tiers.iterrows():
    ax.text(i, r["actual_churn_rate"] + 1, f"{r['actual_churn_rate']:.0f}%\nn={int(r['customers']):,}",
            ha="center", fontsize=8)
ax.axhline(baseline * 100, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("Actual churn rate (%)"); ax.set_title("Actual churn by predicted risk tier")
ax.set_ylim(0, 100); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("chart_risk_tiers.png", dpi=150); plt.close()

# 9. FILE FOR POWER BI -------------------------------------------------------
df["churn_probability"] = df["churn_probability"].round(4)
df.to_csv("churn_scored.csv", index=False)
print("\nSaved churn_scored.csv", df.shape)
