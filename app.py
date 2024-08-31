#import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Setting page config
st.set_page_config(page_title="Stock Market Forecasting")

#title
st.title('Stock Market Forecasting App')
st.subheader('Forecasts the stock market price of the selected company.')
st.image("stock.jpg")  # Assuming this image exists in the deployment environment

#sidebar
st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2024, 1, 1))
end_date = st.sidebar.date_input('End date', date(2024, 8, 29))

#add ticker symbol list
ticker_list = ["AAPL", "MSFT", "GOOG", "FB", "RS", "TSLA", "META", "AMZN", "NVDA", "ADBE", "QCOM", "PYPL", "TCS.NS", "NFLX", "TTM", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

#fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)

#add date as a column to the dataframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write(f'Data from {start_date} to {end_date}')
st.write(data)

#plot the data
st.header('Data Visualization')
fig = px.line(data, x='Date', y=data.columns, title=f'Closing Price of {ticker}', width=800, height=600)
st.plotly_chart(fig)

#add a select box to select column from data
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

#subsetting the data
data = data[["Date", column]]
st.write("Selected Data")
st.write(data)

#ADF test check stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1200, height=400).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1200, height=400).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1200, height=400).update_traces(line_color='Red', line_dash='dot'))

#user input for three parameters of the model and seasonal order
p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

#print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")

#predict the future values (Forecasting)
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions = predictions.predicted_mean

#add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index, True)
predictions.reset_index(drop=True, inplace=True)
st.write("## Predictions", predictions)
st.write("## Actual Data", data)
st.write("---")

#plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions[predictions.columns[1]], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
st.plotly_chart(fig)
