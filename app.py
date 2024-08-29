#import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Setting page config
st.set_page_config(page_title="Stock Market Forecasting")

#title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('Forecasts the stock market price of the selected company.')
st.image("stock.jpg")

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
st.write('Data from', start_date, 'to', end_date)
st.write(data)


#plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select you specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=800, height=600)
st.plotly_chart(fig)

#add a select box to select column from data
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

#subsetting the data
data = data[["Date", column]]
st.write("Selected Data")
st.write(data)

#ADF test check stationarity
st.header('Is data Stationary?')
#st.write('**Note:** If p-value is less than 0.05, then data is stationary')
st.write(adfuller(data[column])[1]<0.05)

#lets decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
#make same plot in plotly
st.write("## Plotting the decomposition in plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

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
#predict the future values
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period)
predictions = predictions.predicted_mean
#st.write(predictions)

#ask for the values of p, d and q from user
p = st.number_input("Enter p value", value=1)
d = st.number_input("Enter d value", value=1)
q = st.number_input("Enter q value", value=2)
seaonal_p = st.number_input("Enter seasonal value", value=12)

model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonl_order=(p, d, q, 12))
model = model.fit(disp=-1)

#print model summary
st.write("## Model summary")
st.write(model.summary())
st.write("---")
#predict the values with user input values
st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data</p>", unsafe_allow_html=True)
forecast_period = st.number_input("## Enter forecast period in days", value=10)

#predict all the values for the forecast period and the current dataset
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions = predictions.predicted_mean
#st.write(len(predictions))
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
#adding actual data to plot
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
#add predicted datan to plot
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
#set title and axis labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
#display the plot
st.plotly_chart(fig)