"""
clean_and_analyze.py
--------------------
Loads the messy covid_raw.csv, cleans it, engineers features,
prints summary statistics, saves covid_cleaned.csv, and plots key charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="darkgrid")

# ── 1. LOAD ─────────────────────────────────────────────────────────────────

print("=" * 55)
print("STEP 1: Loading raw data")
print("=" * 55)

df = pd.read_csv("covid_raw.csv")
print(f"Raw shape: {df.shape}")
print(df.dtypes)
print(df.head())

# ── 2. CLEAN ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2: Cleaning")
print("=" * 55)

# 2a. Remove duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Dropped {before - len(df)} duplicate rows → {len(df)} rows remain")

# 2b. Standardize country names to title case
df["country"] = df["country"].str.strip().str.title()
print("Countries after normalization:", sorted(df["country"].unique()))

# 2c. Parse dates — handle mixed formats
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
bad_dates = df["date"].isna().sum()
print(f"Unparseable dates coerced to NaT: {bad_dates}")
df.dropna(subset=["date"], inplace=True)

# 2d. Sort by country then date
df.sort_values(["country", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)

# 2e. Handle missing numeric values — fill with column median per country
print(f"\nMissing before fill: new_cases={df['new_cases'].isna().sum()}, "
      f"new_deaths={df['new_deaths'].isna().sum()}")

df["new_cases"] = df.groupby("country")["new_cases"].transform(
    lambda s: s.fillna(s.median())
)
df["new_deaths"] = df.groupby("country")["new_deaths"].transform(
    lambda s: s.fillna(s.median())
)
print(f"Missing after fill:  new_cases={df['new_cases'].isna().sum()}, "
      f"new_deaths={df['new_deaths'].isna().sum()}")

# 2f. Cap outliers at 99th percentile per country
def cap_outliers(series, upper_pct=0.99):
    cap = series.quantile(upper_pct)
    return series.clip(upper=cap)

df["new_cases"] = df.groupby("country")["new_cases"].transform(cap_outliers)
df["new_deaths"] = df.groupby("country")["new_deaths"].transform(cap_outliers)
print("Outliers capped at 99th percentile per country.")

# 2g. Enforce non-negative integers
df["new_cases"] = df["new_cases"].clip(lower=0).round().astype(int)
df["new_deaths"] = df["new_deaths"].clip(lower=0).round().astype(int)

print(f"\nClean shape: {df.shape}")
print(df.describe())

# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 3: Feature engineering")
print("=" * 55)

# 7-day rolling average of new cases (per country)
df["cases_7day_avg"] = (
    df.groupby("country")["new_cases"]
    .transform(lambda s: s.rolling(7, min_periods=1).mean())
    .round(1)
)

# Case fatality rate (CFR) — avoid division by zero
df["cfr_pct"] = np.where(
    df["new_cases"] > 0,
    (df["new_deaths"] / df["new_cases"] * 100).round(2),
    np.nan,
)

# Weekly growth rate: % change vs 7 days ago
df["weekly_growth_pct"] = (
    df.groupby("country")["new_cases"]
    .transform(lambda s: s.pct_change(periods=7) * 100)
    .round(2)
)

print(df[["date", "country", "new_cases", "cases_7day_avg", "cfr_pct", "weekly_growth_pct"]].head(20))

# ── 4. SAVE CLEANED DATA ─────────────────────────────────────────────────────

df.to_csv("covid_cleaned.csv", index=False)
print("\nSaved covid_cleaned.csv")

# ── 5. SUMMARY STATS ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 4: Country-level summary")
print("=" * 55)

summary = df.groupby("country").agg(
    total_cases=("new_cases", "sum"),
    total_deaths=("new_deaths", "sum"),
    peak_daily_cases=("new_cases", "max"),
    avg_cfr_pct=("cfr_pct", "mean"),
).round(2)
summary["overall_cfr_pct"] = (summary["total_deaths"] / summary["total_cases"] * 100).round(2)
print(summary.to_string())

# ── 6. VISUALIZATIONS ────────────────────────────────────────────────────────

FOCUS_COUNTRY = "Usa"  # Change to any country in the dataset

country_df = df[df["country"] == FOCUS_COUNTRY].copy()

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
fig.suptitle(f"COVID-19 Trends — {FOCUS_COUNTRY}", fontsize=15, fontweight="bold")

# Chart 1: Daily new cases + 7-day rolling average
ax1 = axes[0]
ax1.bar(country_df["date"], country_df["new_cases"], color="steelblue",
        alpha=0.4, label="Daily new cases")
ax1.plot(country_df["date"], country_df["cases_7day_avg"], color="crimson",
         linewidth=2, label="7-day rolling avg")
ax1.set_title("Daily New Cases with 7-Day Rolling Average")
ax1.set_ylabel("New Cases")
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

# Chart 2: Case fatality rate over time
ax2 = axes[1]
cfr_plot = country_df.dropna(subset=["cfr_pct"])
ax2.plot(cfr_plot["date"], cfr_plot["cfr_pct"], color="darkorange", linewidth=1.5)
ax2.axhline(cfr_plot["cfr_pct"].median(), color="gray", linestyle="--",
            label=f'Median CFR: {cfr_plot["cfr_pct"].median():.2f}%')
ax2.set_title("Daily Case Fatality Rate (%)")
ax2.set_ylabel("CFR (%)")
ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

# Chart 3: Weekly growth rate
ax3 = axes[2]
growth_plot = country_df.dropna(subset=["weekly_growth_pct"])
colors = ["crimson" if x >= 0 else "seagreen" for x in growth_plot["weekly_growth_pct"]]
ax3.bar(growth_plot["date"], growth_plot["weekly_growth_pct"], color=colors, alpha=0.7)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_title("Weekly Growth Rate (% change vs 7 days prior)")
ax3.set_ylabel("Growth Rate (%)")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
plt.savefig("covid_trends.png", dpi=150)
print("\nChart saved to covid_trends.png")
plt.show()

print("\nDone.")