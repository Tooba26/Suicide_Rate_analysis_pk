# Suicide_Rate_analysis_pk
# Pakistan Suicide Rate Analysis & Forecasting

A Jupyter/Colab notebook that analyzes historical suicide rate data for Pakistan (male, female, and overall), visualizes trends, builds a simple machine learning model to forecast future rates, and includes the code for a Streamlit dashboard front end.

## Overview

This notebook walks through a complete mini data-science pipeline:

1. **Data loading & cleaning** — reads three CSV files (male, female, and overall suicide rates) using [Polars](https://www.pola.rs/), a fast Rust-based DataFrame library, and merges them into a single combined dataset keyed by year.
2. **Exploratory analysis & visualization** — line charts of rate trends over time, year-over-year percentage change bar charts, a correlation analysis between male and female rates (with a scatter plot), and grouped bar charts comparing male vs. female rates.
3. **Feature engineering** — builds lagged features (previous 1–2 years) for each rate series to use as predictors.
4. **Modeling** — splits the data chronologically (train on data through 2018, test on 2019+) and fits a multi-output `LinearRegression` model (scikit-learn) to forecast male, female, and overall suicide rates.
5. **Evaluation** — computes MAE, MSE, and R² for each target variable on the test set.
6. **Dashboard** — includes Streamlit code (`app.py`) for an interactive front end to explore the data and model outputs locally.

## Repository Contents

| File | Description |
|---|---|
| `Untitled6.ipynb` | Main notebook containing all analysis, modeling, and dashboard code |
| `Pakistan-Suicide-Rate-Male.csv` | Yearly male suicide rate data (not included — see Data section) |
| `Pakistan-Suicide-Rate-Female.csv` | Yearly female suicide rate data (not included — see Data section) |
| `Pakistan-Suicide-Rate-Suicide-Rate.csv` | Yearly overall suicide rate data (not included — see Data section) |

## Data

The notebook expects three CSV files in the working directory, each with a `Year` column and a `Suicide Rate` column:

- `Pakistan-Suicide-Rate-Male.csv`
- `Pakistan-Suicide-Rate-Female.csv`
- `Pakistan-Suicide-Rate-Suicide-Rate.csv` (overall)

These files are **not bundled** in this repository. Supply your own copies (e.g., from a public health statistics source such as WHO or World Bank) and place them in the same directory as the notebook, or update the file paths in the data-loading cells to match your setup.

## Requirements

- Python 3.10+
- [polars](https://pypi.org/project/polars/)
- matplotlib
- scikit-learn
- numpy
- streamlit (only needed to run the dashboard)

Install everything with:

```bash
pip install polars matplotlib scikit-learn numpy streamlit
```

## Usage

### Run the notebook

1. Clone this repository and place your suicide-rate CSV files alongside the notebook.
2. Open `Untitled6.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells in order. The notebook will:
   - Load and merge the datasets
   - Generate trend, correlation, and comparison charts
   - Train and evaluate the linear regression forecasting model

### Run the dashboard

The notebook includes Streamlit code intended to be saved as a standalone script:

```bash
pip install streamlit
streamlit run app.py
```

## Methodology Notes

- **Forecasting model**: A multi-output linear regression predicts male, female, and overall rates simultaneously from lagged values of each series. This is a baseline approach — results should be interpreted as illustrative rather than authoritative, given the small dataset size typical of yearly national statistics.
- **Train/test split**: Chronological (train ≤ 2018, test > 2018) rather than random, which is appropriate for time-series data but means the test set is small.


**Note on subject matter:** This project analyzes national suicide statistics for research and public-health awareness purposes. If you or someone you know is struggling, please reach out to a crisis line or mental health professional in your country — you don't have to go through it alone.