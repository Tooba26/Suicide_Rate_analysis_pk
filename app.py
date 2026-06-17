import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Pakistan Suicide Rate Analysis & Forecast")

st.title("🇵🇰 Pakistan Suicide Rate Analysis and Forecasting")
st.write("This dashboard visualizes historical suicide rates in Pakistan and provides predictions using a Linear Regression model.")

# --- Data Loading and Preprocessing (Reusing notebook logic) ---
@st.cache_data
def load_data():
    df_female_rate = pl.read_csv("Pakistan-Suicide-Rate-Female.csv")
    df_male_rate = pl.read_csv("Pakistan-Suicide-Rate-Male.csv")
    df_rate_all = pl.read_csv("Pakistan-Suicide-Rate-Suicide-Rate.csv")

    df_female_rate = df_female_rate.rename({'': 'Year', 'Suicide Rate': 'Female Suicide Rate'})
    df_male_rate = df_male_rate.rename({'': 'Year', 'Suicide Rate': 'Male Suicide Rate'})
    df_rate_all = df_rate_all.rename({'': 'Year', 'Suicide Rate': 'Overall Suicide Rate'})

    df_combined = df_male_rate.join(df_female_rate, on='Year', how='inner')
    df_final = df_combined.join(df_rate_all, on='Year', how='inner')

    # Calculate Year-over-Year Percentage Change
    df_final = df_final.with_columns([
        ((pl.col('Male Suicide Rate') - pl.col('Male Suicide Rate').shift(1)) / pl.col('Male Suicide Rate').shift(1) * 100).alias('Male Rate YoY Change %'),
        ((pl.col('Female Suicide Rate') - pl.col('Female Suicide Rate').shift(1)) / pl.col('Female Suicide Rate').shift(1) * 100).alias('Female Rate YoY Change %'),
        ((pl.col('Overall Suicide Rate') - pl.col('Overall Suicide Rate').shift(1)) / pl.col('Overall Suicide Rate').shift(1) * 100).alias('Overall Rate YoY Change %')
    ])

    return df_final.sort('Year')

df_final_dashboard = load_data()
st.subheader("1. Raw Data Overview")
st.write(df_final_dashboard)

# --- Feature Engineering and Model Training (Reusing notebook logic) ---
@st.cache_resource # Cache the model
def train_model(_df_final_data):
    df_model_data = _df_final_data.sort('Year')
    for col in ['Male Suicide Rate', 'Female Suicide Rate', 'Overall Suicide Rate']:
        df_model_data = df_model_data.with_columns(
            pl.col(col).shift(1).alias(f'{col}_Lag1'),
            pl.col(col).shift(2).alias(f'{col}_Lag2')
        )
    df_model_data = df_model_data.drop_nulls()

    split_year = 2018
    train_df = df_model_data.filter(pl.col('Year') <= split_year)
    test_df = df_model_data.filter(pl.col('Year') > split_year)

    features = [
        'Male Suicide Rate_Lag1', 'Male Suicide Rate_Lag2',
        'Female Suicide Rate_Lag1', 'Female Suicide Rate_Lag2',
        'Overall Suicide Rate_Lag1', 'Overall Suicide Rate_Lag2'
    ]
    targets = ['Male Suicide Rate', 'Female Suicide Rate', 'Overall Suicide Rate']

    X_train = train_df.select(features)
    y_train = train_df.select(targets)
    X_test = test_df.select(features)
    y_test = test_df.select(targets)

    model = LinearRegression()
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # Make predictions on the test set
    y_pred = model.predict(X_test.to_numpy())
    y_pred_df = pl.DataFrame(y_pred, schema=y_test.schema)
    y_pred_df = y_pred_df.rename({
        'Male Suicide Rate': 'Male Suicide Rate_Predicted',
        'Female Suicide Rate': 'Female Suicide Rate_Predicted',
        'Overall Suicide Rate': 'Overall Suicide Rate_Predicted'
    })

    return model, X_test, y_test, y_pred_df, df_model_data, test_df # Return test_df here

model, X_test, y_test, y_pred_df, df_model_data, test_df = train_model(df_final_dashboard)

# --- Visualizations ---
st.subheader("2. Suicide Rate Trends Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_final_dashboard['Year'], df_final_dashboard['Male Suicide Rate'], label='Male Suicide Rate', marker='o')
ax.plot(df_final_dashboard['Year'], df_final_dashboard['Female Suicide Rate'], label='Female Suicide Rate', marker='x')
ax.plot(df_final_dashboard['Year'], df_final_dashboard['Overall Suicide Rate'], label='Overall Suicide Rate', marker='s')
ax.set_title('Suicide Rates Over Time in Pakistan (2000-2021)')
ax.set_xlabel('Year')
ax.set_ylabel('Suicide Rate')
ax.grid(True)
ax.legend()
ax.set_xticks(df_final_dashboard['Year'].to_list())
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

st.subheader("3. Male vs. Female Suicide Rates (Bar Chart)")
fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.35
r1 = np.arange(len(df_final_dashboard['Year']))
r2 = [x + bar_width for x in r1]

ax.bar(r1, df_final_dashboard['Male Suicide Rate'], color='skyblue', width=bar_width, label='Male Suicide Rate')
ax.bar(r2, df_final_dashboard['Female Suicide Rate'], color='salmon', width=bar_width, label='Female Suicide Rate')

ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Suicide Rate', fontweight='bold')
ax.set_title('Male vs. Female Suicide Rates Over Time', fontweight='bold')
ax.set_xticks([r + bar_width / 2 for r in r1], df_final_dashboard['Year'].to_list(), rotation=45)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

st.subheader("4. Year-over-Year Percentage Change in Suicide Rates")
df_yoy_change_dashboard = df_final_dashboard.drop_nulls()
fig, ax = plt.subplots(figsize=(14, 8))
width = 0.2
x = np.arange(len(df_yoy_change_dashboard['Year']))
ax.bar(x - width, df_yoy_change_dashboard['Male Rate YoY Change %'].to_numpy(), width, label='Male Rate YoY Change %', color='skyblue')
ax.bar(x, df_yoy_change_dashboard['Female Rate YoY Change %'].to_numpy(), width, label='Female Rate YoY Change %', color='salmon')
ax.bar(x + width, df_yoy_change_dashboard['Overall Rate YoY Change %'].to_numpy(), width, label='Overall Rate YoY Change %', color='lightgreen')
ax.set_title('Year-over-Year Percentage Change in Suicide Rates')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage Change (%)')
ax.set_xticks(x, df_yoy_change_dashboard['Year'].to_list(), rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()
st.pyplot(fig)

st.subheader("5. Relationship Between Male and Female Suicide Rates (Scatter Plot)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_final_dashboard['Male Suicide Rate'], df_final_dashboard['Female Suicide Rate'])
ax.set_title('Relationship Between Male and Female Suicide Rates')
ax.set_xlabel('Male Suicide Rate')
ax.set_ylabel('Female Suicide Rate')
ax.grid(True)
st.pyplot(fig)


st.subheader("6. Model Predictions vs. Actuals")
st.write("Here are the actual suicide rates and the model's predictions for the test set (years after 2018).")

col1, col2 = st.columns(2)
with col1:
    st.write("**Actual Test Data (y_test)**")
    st.dataframe(y_test)
with col2:
    st.write("**Predicted Test Data (y_pred)**")
    st.dataframe(y_pred_df)

# Combine actual and predicted for visualization
# Add 'Year' column to y_test and y_pred_df using the 'Year' column from test_df
y_test_with_year = y_test.with_columns(test_df.select('Year'))
y_pred_df_with_year = y_pred_df.with_columns(test_df.select('Year'))

comparison_df = pl.concat([
    y_test_with_year.with_columns(pl.lit('Actual').alias('Type')),
    y_pred_df_with_year.rename({
        'Male Suicide Rate_Predicted': 'Male Suicide Rate',
        'Female Suicide Rate_Predicted': 'Female Suicide Rate',
        'Overall Suicide Rate_Predicted': 'Overall Suicide Rate'
    }).with_columns(pl.lit('Predicted').alias('Type'))
]).sort('Year', 'Type')

st.write("#### Visual Comparison")
for rate_type in ['Male Suicide Rate', 'Female Suicide Rate', 'Overall Suicide Rate']:
    st.write(f"##### {rate_type} - Actual vs. Predicted")
    fig, ax = plt.subplots(figsize=(10, 5))
    actual = comparison_df.filter(pl.col('Type') == 'Actual')
    predicted = comparison_df.filter(pl.col('Type') == 'Predicted')

    ax.plot(actual['Year'], actual[rate_type], marker='o', label='Actual')
    ax.plot(predicted['Year'], predicted[rate_type], marker='x', linestyle='--', label='Predicted')
    ax.set_title(f'{rate_type} (Actual vs. Predicted)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Suicide Rate')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


st.subheader("7. Deploying Your Dashboard")
st.write("You can deploy this Streamlit app for free using [Streamlit Community Cloud](https://streamlit.io/cloud). Just save this code as `app.py`, push it to a GitHub repository, and connect it to your Streamlit Cloud account!")
