"""
generate_data.py
----------------
Generates a realistic but intentionally messy COVID-19 CSV dataset.
Introduces: missing values, duplicate rows, inconsistent date formats,
mixed-case country names, and outlier spikes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# --- Parameters ---
N_DAYS = 365
START_DATE = "2020-03-01"
COUNTRIES = ["USA", "Germany", "Brazil", "India", "france"]  # intentional mixed case

records = []

for country in COUNTRIES:
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

    # Simulate a wave pattern using sine curves
    t = np.linspace(0, 4 * np.pi, N_DAYS)
    base_cases = (np.sin(t) + 1.1) * 5000 + np.random.normal(0, 400, N_DAYS)
    base_cases = np.clip(base_cases, 0, None).astype(int)

    # Deaths roughly 1-3% of cases with lag
    base_deaths = (base_cases * np.random.uniform(0.01, 0.03, N_DAYS)).astype(int)

    for i, date in enumerate(dates):
        # Introduce missing values (~5%)
        cases = base_cases[i] if np.random.rand() > 0.05 else np.nan
        deaths = base_deaths[i] if np.random.rand() > 0.05 else np.nan

        # Introduce outlier spikes (~1%)
        if np.random.rand() < 0.01 and not np.isnan(cases):
            cases = cases * np.random.uniform(8, 15)

        # Introduce inconsistent date formats for ~10% of rows
        if np.random.rand() < 0.10:
            date_str = date.strftime("%m/%d/%Y")
        else:
            date_str = date.strftime("%Y-%m-%d")

        records.append({
            "date": date_str,
            "country": country,
            "new_cases": cases,
            "new_deaths": deaths,
        })

df = pd.DataFrame(records)

# Inject ~30 duplicate rows
duplicate_sample = df.sample(30, random_state=7)
df = pd.concat([df, duplicate_sample], ignore_index=True)

# Shuffle the rows so dates are out of order
df = df.sample(frac=1, random_state=99).reset_index(drop=True)

df.to_csv("covid_raw.csv", index=False)
print(f"Saved covid_raw.csv — {len(df)} rows, {df.shape[1]} columns")
print("Sample:\n", df.head(10))