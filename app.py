import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import date

# Setting page config
st.set_page_config(page_title="Stock Market Forecasting")

# Title and Header
st.title('Stock Market Forecasting App')
st.subheader('Forecasts the stock market price of the selected company.')
st.image("stock.jpg")

# Sidebar for input
st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2024, 1, 1))
end_date = st.sidebar.date_input('End date', date(2024, 8, 29))

# Add ticker symbol list
ticker_list = ["AAPL", "MSFT", "GOOG", "FB", "RS", "TSLA", "META", "AMZN", "NVDA", "ADBE", "QCOM", "PYPL", "TCS.NS", "NFLX", "TTM", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch data from yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Add date as a column to the dataframe
data.insert(0, "Date", data.index)
data.reset_index(drop=True, inplace=True)

# Display data
st.write(f'Data from {start_date} to {end_date}')
st.write(data)

# Plot the data
st.header('Data Visualization')
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=800, height=600)
st.plotly_chart(fig)

# Add a select box to select column from data
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

# Subsetting the data
data = data[["Date", column]]
st.write("Selected Data")
st.write(data)

# Convert Date column to datetime format and column to numeric
data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0)

# ADF test check stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
st.write("## Plotting the decomposition in Plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend').update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality').update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals').update_traces(line_color='Red'))

# SARIMAX model parameters
p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

# Fit SARIMAX model
model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
model = model.fit()

# Display model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")

# Forecast future values
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period - 1)
predicted_mean = predictions.predicted_mean

# Convert predictions to DataFrame with Dates
predictions_df = pd.DataFrame(predicted_mean)
predictions_df.index = pd.date_range(start=end_date, periods=len(predicted_mean), freq='D')
predictions_df.reset_index(inplace=True)
predictions_df.columns = ['Date', 'Predicted']

# Display predictions and actual data
st.write("## Predictions", predictions_df)
st.write("## Actual Data", data)
st.write("---")

# Plot Actual vs Predicted data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions_df["Date"], y=predictions_df["Predicted"], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
st.plotly_chart(fig)
