![delta](https://github.com/user-attachments/assets/0c968227-0f5e-4f95-9ebc-e72e2414e4ef)

# Predicting the Future Stock Price of Delta Airlines

📈 The stock market is volatile and unpredictable, and single stocks are especially notoriously difficult to predict. 
Stock prices change from day to day, and trying to predict them is no easier than predicting the future 🔮. 
However, the price of tomorrow is undoubtedly dependent on today's price, yesterday's price, and so forth.
Using a univariate time series model to predict stock prices is returning to the basics--it allows for a clean and focused approach. 

In this project, I will use time series models to predict the future stock price of Delta Airlines (DAL) ✈️.

### Process Overview

1. [Get Data](#Getting-the-Data)
2. [Check Data for Stationarity](#Check-for-Stationarity)
3. [Modeling](#Modeling)
     1. ARIMA with hyperparameters selected using ACF and PACF plots
     2. ARIMA with hyperparameters selected through autoARIMA
     3. ARIMA with hyperparameters selected through a loop search
     4. Prophet with hyperparameters selected through a loop search
4. [Predicting](#Predicting)

### Getting the Data

Stock prices are obtained from **yfinance's** api. Documentation can be found here: [yfinance documentation](http://yfinance-python.org/)
The yfinance api allows for data to be downloaded using the stock's ticker symbol, and for a period of time as defined by a start date and end date.
Delta Airline's ticker symbol is **DAL**, and my data was from the period between **2022-04-01** to **2025-03-31**. 
The api returns the Close, High, Low, Open prices as well as the Volume. My time series model utilizes past **Close** prices to predict future Close prices.
In order to conduct time series modeling on the data, the data was cleaned using the following steps:

1. Ensure the Date is in datetime format
2. Set the Date as the index of the dataframe
3. Set the frequency to be 'B' for business days
4. Business days does not account for holidays, so forward-fill null values with the price from the previous period that has a value

### Check for Stationarity

Data must be stationary before applying an ARIMA model and using ACF and PACF plots to select the appropriate hyperparameters (p, d, and q). ARIMA assumes stationarity — meaning the statistical properties of the series do not change over time. Data is considered stationary if the following conditions are met:

1. **Constant mean** - the average value remains stable over time. A trending mean (either upward or downward) indicates non-stationarity. Stationary data should show no long-term trend.
2. **Constant variance (homoscedasticity)** - variability (spread) of the data should be consistent throughout the series. If the variance changes over time (e.g., increases during certain periods), the series is heteroscedastic and not stationary.
3. **Autocovariance** that does not depend on time - the relationship between values at different lags should depend only on the lag distance, not on the actual time at which the series is observed. Patterns such as seasonality — where values repeat at regular intervals — introduce time-dependent autocovariance and must be removed or modeled to achieve stationarity.

To check the data stationary, I will plot the rolling mean and standard deviation and looked for signs of non-constant mean, non-constant variance, and cyclical patterns in the ups and downs.

![rolling_mean_std](https://github.com/user-attachments/assets/31833b64-3eb1-4a2a-a56e-5bf318bbf890)

Observations of the above plot:

- The rolling mean shows a positive trend in the close price
- The rolling standard deviation also signals that the standard deviation is increasing over time
- The ups and downs show signs of cyclicality, but the patterns are nto well defined

Based on the above, the data is not stationary. I will confirm this by using the ADFuller test. The null hypothesis of the ADFuller test is that the data is not stationary, I would need a p value of less 0.05 in order to reject my null hypothesis.



### Modeling

### Predicting



