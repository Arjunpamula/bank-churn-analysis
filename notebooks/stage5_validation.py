import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("churn_clean.csv")
baseline = df["churn"].mean()

# ---------------------------------------------------------------------------
# TEST 1: Is each factor related to churn at all? (chi-square test)
# Null hypothesis: churn rate is the SAME in every group.
# p-value = chance of seeing differences this big if that were true.
# Cramer's V = how STRONG the link is (0 = none, ~0.1 small, ~0.3 medium, 0.5+ large).
# ---------------------------------------------------------------------------
factors = ["age_group", "product_group", "country", "active_label",
           "gender", "balance_band", "tenure_band", "credit_card_label"]
rows = []
for f in factors:
    table = pd.crosstab(df[f], df["churn"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    rows.append({"factor": f, "chi2": round(chi2, 1), "dof": dof, "p_value": p, "cramers_v": round(v, 3)})

t1 = pd.DataFrame(rows)
# Correct for running 8 tests at once (Holm method)
t1["p_adjusted"] = multipletests(t1["p_value"], method="holm")[1]
t1["significant_at_5%"] = t1["p_adjusted"] < 0.05
t1["p_value"] = t1["p_value"].map(lambda x: "<1e-300" if x < 1e-300 else f"{x:.2e}")
t1["p_adjusted"] = t1["p_adjusted"].map(lambda x: "<1e-300" if x < 1e-300 else f"{x:.2e}")


def strength(v):
    return "none" if v < 0.03 else "weak" if v < 0.1 else "moderate" if v < 0.3 else "strong"


t1["strength"] = t1["cramers_v"].map(strength)
t1 = t1.sort_values("cramers_v", ascending=False)
print("=== TEST 1: Chi-square, each factor vs churn ===")
print(t1.to_string(index=False))

# ---------------------------------------------------------------------------
# TEST 2: How precise are the key segment rates? (95% confidence intervals)
# A small group gives a wide interval = less trustworthy rate.
# ---------------------------------------------------------------------------
segments = {
    "Age 51-60": df["age_group"] == "51-60",
    "Age 41-50": df["age_group"] == "41-50",
    "Germany": df["country"] == "Germany",
    "Inactive": df["active_label"] == "Inactive",
    "1 product": df["product_group"] == "1 product",
    "3+ products": df["product_group"] == "3+ products",
    "1 product + Inactive": (df["product_group"] == "1 product") & (df["active_label"] == "Inactive"),
    "Age 51-60 + Inactive": (df["age_group"] == "51-60") & (df["active_label"] == "Inactive"),
    "Germany + 1 product": (df["country"] == "Germany") & (df["product_group"] == "1 product"),
    "3+ products + Active": (df["product_group"] == "3+ products") & (df["active_label"] == "Active"),
}
rows = []
for name, mask in segments.items():
    n = int(mask.sum())
    k = int(df.loc[mask, "churn"].sum())
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    rows.append({"segment": name, "customers": n, "churn_rate_%": round(k / n * 100, 1),
                 "CI_95_low_%": round(lo * 100, 1), "CI_95_high_%": round(hi * 100, 1),
                 "CI_width_pts": round((hi - lo) * 100, 1),
                 "clearly_above_baseline": lo > baseline})
t2 = pd.DataFrame(rows)
print(f"\n=== TEST 2: 95% confidence intervals for key segments (baseline {baseline*100:.1f}%) ===")
print(t2.to_string(index=False))

# ---------------------------------------------------------------------------
# TEST 3: Is the Germany effect REAL or just because Germans are older/inactive?
# Cochran-Mantel-Haenszel: compares Germany vs rest WITHIN each age x activity
# stratum, then pools. Odds ratio > 1 with CI excluding 1 = real, independent effect.
# ---------------------------------------------------------------------------
df["is_germany"] = (df["country"] == "Germany").astype(int)
tables = []
for _, g in df.groupby(["age_group", "active_label", "product_group"], observed=True):
    t = pd.crosstab(g["is_germany"], g["churn"]).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
    if t.values.min() >= 0 and t.values.sum() > 0 and (t.sum(axis=1) > 0).all() and (t.sum(axis=0) > 0).all():
        tables.append(t.values)
st = StratifiedTable(tables)
lo, hi = st.oddsratio_pooled_confint()
print("\n=== TEST 3: Germany effect after controlling for age, activity and products ===")
print(f"Crude odds ratio (no control):   {StratifiedTable([pd.crosstab(df['is_germany'], df['churn']).reindex(index=[1,0], columns=[1,0]).values]).oddsratio_pooled:.2f}")
print(f"Adjusted odds ratio (CMH pooled): {st.oddsratio_pooled:.2f}  (95% CI {lo:.2f} to {hi:.2f})")
print(f"Homogeneity test p-value: {st.test_equal_odds().pvalue:.3f}")
print("-> Germany remains a risk factor after control" if lo > 1 else "-> Germany effect not clearly independent")

# ---------------------------------------------------------------------------
# TEST 4: Does inactivity matter MORE for older customers? (interaction test)
# Compare a logistic model with and without the age x activity interaction.
# ---------------------------------------------------------------------------
df["older"] = df["age"].ge(50).astype(int)
m_simple = smf.logit("churn ~ older + active_member", data=df).fit(disp=0)
m_inter = smf.logit("churn ~ older * active_member", data=df).fit(disp=0)
lr = 2 * (m_inter.llf - m_simple.llf)
p_lr = stats.chi2.sf(lr, df=1)
print("\n=== TEST 4: Interaction, is inactivity effect different for age 50+? ===")
print(f"Likelihood-ratio statistic: {lr:.1f}, p-value: {p_lr:.2e}")
print("-> Yes, the effect of inactivity differs by age" if p_lr < 0.05 else "-> No clear interaction")

# ---------------------------------------------------------------------------
# TEST 5: How much do the small groups worry us? Rule of thumb check.
# ---------------------------------------------------------------------------
print("\n=== TEST 5: Group-size check for 3+ products ===")
g3 = df[df["product_group"] == "3+ products"]
print(f"3+ product customers: {len(g3)} ({len(g3)/len(df):.1%} of all), "
      f"holding {g3['churn'].sum()/df['churn'].sum():.1%} of all churners")
print("4-product group alone:", int((df['products_number'] == 4).sum()), "customers")

t1.to_csv("validation_chi_square.csv", index=False)
t2.to_csv("validation_confidence_intervals.csv", index=False)
