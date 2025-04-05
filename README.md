![delta](https://github.com/user-attachments/assets/0c968227-0f5e-4f95-9ebc-e72e2414e4ef)

# Predicting the Future Stock Price of Delta Airlines

📈 The stock market is volatile and unpredictable, and single stocks are especially notoriously difficult to predict. 
Stock prices change from day to day, and trying to predict them is no easier than predicting the future 🔮. 
However, the price of tomorrow is undoubtedly dependent on today's price, yesterday's price, and so forth.
Using a univariate time series model to predict stock prices is returning to the basics--it allows for a clean and focused approach. 

In this project, I use time series models to predict the future stock price of Delta Airlines (DAL) ✈️.

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

##### Plotting the rolling mean and standard deviation

To check the data stationary, I plotted the rolling mean and standard deviation and looked for signs of non-constant mean, non-constant variance, and cyclical patterns in the ups and downs.

![rolling_mean_std](https://github.com/user-attachments/assets/31833b64-3eb1-4a2a-a56e-5bf318bbf890)

My observations of the above plot were as follows: 

- The rolling mean shows a positive trend in the close price
- The rolling standard deviation also signals that the standard deviation is increasing over time
- The ups and downs show signs of cyclicality, but the patterns are nto well defined

Therefore, the data is not stationary. This can be confirmed using the ADFuller test. 

##### Using the ADFuller test to check for stationarity.

The null hypothesis of the ADFuller test is that the data is not stationary, I would need a p value of less 0.05 in order to reject my null hypothesis. The ADFuller test returned a p-value of 0.5958, which is much greater than 0.05. This means that I cannot reject the null hypothesis that the data is not stationary. 

##### Transform the data to become more stationary.

To make the data stationary, I removed non-constant variance by taking the log the close price, and then removed the trend by taking the difference of the logged close price. After logging the data, and taking 1 level of difference, again, I plotted the data along with the rolling mean and standard deviation to check for stationarity. 

![log_diff_rolling_mean_std](https://github.com/user-attachments/assets/4c17a25d-dbbf-4f62-9449-8d06d6f98448)

My observations of the above plot were as follows:

 - The rolling mean no longer has an upward trend
 - The standard deviation shows random variations and does not drift consistenty up or down over time
 - There are no obvious patterns of cyclicality in the transformed data

To confirm that the data is now stationary, again, I checked the ADFuller test's p-value. The ADFuller test on the data transformed by taking the log, and then taken 1 level of difference returned a p-value of 2.33e-30, which is much smaller that 0.05, so I can confidently reject the null hypothesis that the data is non-stationary. 

##### Check for Seasonality.

In addition to flattening trends, differencing also helps to alleviate seasonality, but since the ADFuller test does not check for seasonality, I needed to double check that there is no more seasonality left in the transformed data.  

1. First, I used seasonal decompose to visualize seasonality:

     ![seasonal_decompose](https://github.com/user-attachments/assets/49c1dc33-10f5-4633-a1ac-88751e7fcd48)

     My observations of the above plot were as follows:
     
     1. The top plot is the transformed Data.
     2. The second plot visualizes if there are any trend. Since it is flat on average, and fluctuating randomly, it suggests that there are no strong long-term trend.
     3. The third plot addresses seasonality, where repeating patterns indicate the prescence of seasonality. The plot shows tight, high-frequency seasonal pattern. This could be true seasonality *but* it can also be the result of noise and market micro-patterns. I dived into this further in the next step.
     4. The fourth plot plots the residuals. There are no clear patterns, and the residuals are randomly spread out around 0, so the data does not display signs of being non-stationary.

2. To conduct a deeper investigation of seasonality, I calculated the period of seasonality using periodogram. Periodogram from scipy.signal can help determine how much of the data's variance is explained by each frequency, where the frequencies are an array of the most dominant frequency values. When 1 is divided by the frequency, we get the approximate cycle length of the repeating cycle. For the periodogram computation, I chose to use the data *after* log transformation, but *without differencing*. This is because differencing can weaken the seasonal signals and/or distort them. During the modeling process later, I will need to know seasonal period in order to remove seasonality.

     1. Using the periodogram method, I found 2 dominant periods, 781 and 195.25 business days.
     2. The data contains 781 periods, so this indicates that there is still an overarching trend, which will categorize the entire dataset as 1 cycle. The trend is as expected, and will be removed through differencing.
     3. The second dominant period of 195.25 is likely to be a true seasonal cycle. To put this period into perspective, 195.25 business days translate to roughly 275 days on the standard calendar, which is 3 quarters of a year.
     4. The below plot also shows the peak in the periodogram at 195.25. After 260, the power continues to increase, this is likely due to a trend in the data.

     ![periodogram](https://github.com/user-attachments/assets/6ab96bbc-880c-4716-94cd-beab5ffde17a)


### Modeling

### Predicting



