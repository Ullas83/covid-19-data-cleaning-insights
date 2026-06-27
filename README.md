# COVID-19 Data Cleaning + Insights

A beginner-to-intermediate Python/Pandas project that simulates a real-world data cleaning workflow using a COVID-19 dataset, computes trend metrics, and produces visualizations.

## Project Overview

This project demonstrates:
- **Data Cleaning**: Handling missing values, fixing data types, removing duplicates, and standardizing columns
- **Feature Engineering**: Computing 7-day rolling averages, case fatality rates, and weekly growth rates
- **Data Visualization**: Plotting trends with matplotlib and seaborn

## Project Structure

```
covid-insights/
├── README.md
├── generate_data.py      # Generates a realistic messy sample dataset
├── clean_and_analyze.py  # Main pipeline: cleaning + analysis + charts
├── requirements.txt      # Python dependencies
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/covid-insights.git
cd covid-insights
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample dataset

```bash
python generate_data.py
```

This creates `covid_raw.csv` in the project directory.

### 4. Run the analysis pipeline

```bash
python clean_and_analyze.py
```

This will:
- Print cleaning steps and summary stats to the console
- Save cleaned data to `covid_cleaned.csv`
- Display three charts

## Key Findings (Sample Data)

- **Daily new cases** show a clear wave pattern with peaks roughly every 60–80 days.
- **7-day rolling averages** smooth out weekend reporting dips and reveal the true trend.
- **Case fatality rate (CFR)** starts elevated early in the dataset (data artifacts / early underreporting) and stabilizes between 1–3%.
- **Weekly growth rate** identifies acceleration and deceleration phases better than raw counts.

## Skills Demonstrated

| Skill | Where |
|---|---|
| Python | Both scripts |
| Pandas (cleaning, groupby, rolling) | `clean_and_analyze.py` |
| NumPy | `generate_data.py` |
| Matplotlib / Seaborn | `clean_and_analyze.py` |
| Data documentation | This README |

## License

MIT