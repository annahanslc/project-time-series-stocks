![delta](https://github.com/user-attachments/assets/0c968227-0f5e-4f95-9ebc-e72e2414e4ef)

# Predicting the Future Stock Price of Delta Airlines

📈 The stock market is volatile and unpredictable, and single stocks are especially notoriously difficult to predict. 
Stock prices change from day to day, and trying to predict them is no easier than predicting the future 🔮. 
However, the price of tomorrow is undoubtedly dependent on today's price, yesterday's price, and so forth.
Using a univariate time series model to predict stock prices is returning to the basics--it allows for a clean and focused approach. 

In this project, I use time series models to predict the future stock price of Delta Airlines (DAL) ✈️.

# Process Overview

1. [Get Data](#Getting-the-Data)
2. [Check Data for Stationarity](#Check-for-Stationarity)
3. [Modeling](#Modeling)
     1. ARIMA with hyperparameters selected using ACF and PACF plots
     2. ARIMA with hyperparameters selected through AutoARIMA
     3. ARIMA with hyperparameters selected through a loop search
     4. Prophet with hyperparameters selected through a loop search
4. [Predicting](#Predicting)

# Getting the Data

Stock prices are obtained from **yfinance's** api. Documentation can be found here: [yfinance documentation](http://yfinance-python.org/)
The yfinance api allows for data to be downloaded using the stock's ticker symbol, and for a period of time as defined by a start date and end date.
Delta Airline's ticker symbol is **DAL**, and my data was from the period between **2022-04-01** to **2025-03-31**. 
The api returns the Close, High, Low, Open prices as well as the Volume. My time series model utilizes past **Close** prices to predict future Close prices.
In order to conduct time series modeling on the data, the data was cleaned using the following steps:

1. Ensure the Date is in datetime format
2. Set the Date as the index of the dataframe
3. Set the frequency to be 'B' for business days
4. Business days does not account for holidays, so forward-fill null values with the price from the previous period that has a value

# Check for Stationarity

Data must be stationary before applying an ARIMA model and using ACF and PACF plots to select the appropriate hyperparameters (p, d, and q). ARIMA assumes stationarity — meaning the statistical properties of the series do not change over time. Data is considered stationary if the following conditions are met:

1. **Constant mean** - the average value remains stable over time. A trending mean (either upward or downward) indicates non-stationarity. Stationary data should show no long-term trend.
2. **Constant variance (homoscedasticity)** - variability (spread) of the data should be consistent throughout the series. If the variance changes over time (e.g., increases during certain periods), the series is heteroscedastic and not stationary.
3. **Autocovariance** that does not depend on time - the relationship between values at different lags should depend only on the lag distance, not on the actual time at which the series is observed. Patterns such as seasonality — where values repeat at regular intervals — introduce time-dependent autocovariance and must be removed or modeled to achieve stationarity.

### Plotting the rolling mean and standard deviation

To check the data stationary, I plotted the rolling mean and standard deviation and looked for signs of non-constant mean, non-constant variance, and cyclical patterns in the ups and downs.

![rolling_mean_std](https://github.com/user-attachments/assets/31833b64-3eb1-4a2a-a56e-5bf318bbf890)

My observations of the above plot were as follows: 

- The rolling mean shows a positive trend in the close price
- The rolling standard deviation also signals that the standard deviation is increasing over time
- The ups and downs show signs of cyclicality, but the patterns are nto well defined

Therefore, the data is not stationary. This can be confirmed using the ADFuller test. 

### Using the ADFuller test to check for stationarity.

The null hypothesis of the ADFuller test is that the data is not stationary, I would need a p value of less 0.05 in order to reject my null hypothesis. The ADFuller test returned a p-value of 0.5958, which is much greater than 0.05. This means that I cannot reject the null hypothesis that the data is not stationary. 

### Transform the data to become more stationary.

To make the data stationary, I removed non-constant variance by taking the log the close price, and then removed the trend by taking the difference of the logged close price. After logging the data, and taking 1 level of difference, again, I plotted the data along with the rolling mean and standard deviation to check for stationarity. 

![log_diff_rolling_mean_std](https://github.com/user-attachments/assets/4c17a25d-dbbf-4f62-9449-8d06d6f98448)

My observations of the above plot were as follows:

 - The rolling mean no longer has an upward trend
 - The standard deviation shows random variations and does not drift consistenty up or down over time
 - There are no obvious patterns of cyclicality in the transformed data

To confirm that the data is now stationary, again, I checked the ADFuller test's p-value. The ADFuller test on the data transformed by taking the log, and then taken 1 level of difference returned a p-value of 2.33e-30, which is much smaller that 0.05, so I can confidently reject the null hypothesis that the data is non-stationary. 

### Check for Seasonality.

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
     5. I will be using 195.25 days as the length of a period in my modeling later.

     ![periodogram](https://github.com/user-attachments/assets/6ab96bbc-880c-4716-94cd-beab5ffde17a)


# Modeling

I explored the following models for my time series:

1. ARIMA with hyperparameters selected using ACF and PACF plots
2. ARIMA with hyperparameters selected through AutoARIMA
3. ARIMA with hyperparameters selected through a looped search
4. Prophet with hyperparameters selected through a looped search
     
### 1. ARIMA with hyperparameters selected using ACF and PACF plots

To stablize the data, I performed a log transformation, took 2 levels of difference, and removing seasonality by subtracting seasonal from seasonal_decompose. The resulting p-value from ADFuller is 6.783e-19, which means I can confidently reject the null hypothesis that the data is non-stationary.

To manual determine the best hyperparameters for my ARIMA model, I generated the following ACF and PACF plots:

<img src="https://github.com/user-attachments/assets/91eef2f7-9366-4192-8f84-2301413eaa6c" width="400">
<img src="https://github.com/user-attachments/assets/575a3286-7324-42d4-848a-f30807d1262e" width="400">

My observations of the above plots were as follows:

- The ACF plot shows a drop in correlation after lag of 1, which suggests that the MA component in the ARIMA model is 1 (q=1)
- The PACF plot shows that after lag of 8, the partial autocorrelation becomes insignificant. This suggests that the AR component of the model is 8 (p=8).

The ARIMA model can perform differencing, so the data I used for training the ARIMA model was transformed by taking the log, and deseasonalized by subtracting seasonal from seasonal_decompose, but not yet differenced. In the model, I set d=2 to implement 2 levels of differencing.

My first ARIMA model with hyperparameters of p=8, d=2, q=1, produced the following metrics:

1. Train RMSE = 7.737
2. Test RMSE = 9.475

The predictions for test are plotted alongside the actual values in the below plot:

![arima821_preds](https://github.com/user-attachments/assets/7d25c0f0-58b4-4473-a8a3-721984722816)

The prediction is overall linear. It captures, on average, the actual values on average, but it misses the nuances in the ups and downs. The model is not bad, but I will continue to try to improve my model.

### 2. ARIMA with hyperparameters selected through AutoARIMA

The function AutoARIMA(), from the pmdarima library, will automatically find the best ARIMA or SARIMA (ARIMA with seasonality) model for a time series. It searches over a combination of p, d, q and seasonal P, D, Q to find the best model based on metrics, primarily  AIC and BIC. By default, it will select the model with the lowest AIC score. 

Using AutoARIMA with the following hyperparameter ranges, with both the starting and ending numbers included:

p from 0 to 10 
d from 0 to 3
q from 0 to 5

The best model determined by the AutoARIMA is q=0, d=1, p=1. This is a very simple model, and it basically just uses the previous period's value to predict future values. The predictions for the test timeframe can be seens as a straight line in the plot below:

![autoairma_preds](https://github.com/user-attachments/assets/20cd36ce-60f2-4de9-8542-5c28b4ec47fd)

Predictions performance is as follows:

1. Train RMSE = 1.651
2. Test RMSE = 8.260

The performance is indeed better than my previous model. However, it only considers the previous period's value in its prediction. It does not take into any autocorrelation, or moving average. This makes it a poor model for predicting any longer periods of time. 

As I mentioned eariler, the AutoARIMA() function uses the AIC score in its best model selection. I am more concerned about the model's ability to predict the test period, so test RMSE is the most important metric in my evaluation. Therefore, I will continue to improve my model with this as the goal.

### 3. ARIMA with hyperparameters selected through a manual looped search

To find the combination of hyperparameters that will return the lowest test RMSE, I created a function to return the train and test RMSE for an ARIMA model with p, d and q as arguments. Next, I created a loop using product (from itertools) to try all combinations of p, d, and q from a set range of values in the function. The RMSE scores are then appended to a dataframe that collects all the scores. 

After the loop has run. The dataframe is sorted by ascending test RMSE score, so that the lowest is on top. The top 10 models are:

|   p |   d |   q |   train_rmse |   test_rmse |
|----:|----:|----:|-------------:|------------:|
|   5 |   2 |   1 |      7.7353  |     7.7969  |
|   0 |   2 |   0 |      7.78225 |     7.9499  |
|   3 |   1 |   4 |      1.6502  |     8.2429  |
|   3 |   1 |   3 |      1.64986 |     8.24903 |
|   4 |   1 |   4 |      1.64992 |     8.25278 |
|   0 |   1 |   0 |      1.65084 |     8.25956 |
|   0 |   1 |   1 |      1.65083 |     8.25959 |
|   1 |   1 |   0 |      1.65084 |     8.25963 |
|   4 |   1 |   0 |      1.65058 |     8.26214 |
|   4 |   1 |   1 |      1.65058 |     8.26218 |

Using the hyperparameters that produced the best model, predictions were made for the test timeframe and plotted against actual values below:

![arima_loop](https://github.com/user-attachments/assets/46a864dd-ad7a-44bd-a080-cb72b2a9b247)

The predictions are, again, a straight line, but the test RMSE is the best so far. Again, the model will not fair well when predicting longer periods of time, so I will continue to look for a better model.

### 4. Prophet with hyperparameters selected through a manual looped search

Prophet is a time series forecasting model that was developed by Facebook. It is able to decompose time series into trend, seasonality, holiday effects, and noise. A Prophet model has several hyperparameters that can be tuned, the two that I will search for are the following:

1. **changepoint_prior_scale** controls how stable the trend line is. A lower value is good for when the data has noise and we just want to capture the overall direction. A high value means that the model will allow sudden jumps in trend, so that it can account for real disruptions. 
2. **seasonality_prior_scale** controls the strength of the seasonality. A lower values means that the seasonality is smooth and simple, while a high value will allow for more complex seasonal patterns.

In addition, Prophet allows for a custom seasonality period to be defined using the method **.add_seasonality**. The Prophet default seasonality are daily, weekly, and yearly. Previously, I found that the seasonality in the data has a period length of 195.25, so I used this custom seasonality in my Prophet model. 

The .add_seasonality method also accepts the parameter **fourier_order**, which will tell the model how many Fourier terms to use to model the seasonal pattern. The fourier_order controls how complex the seasonal component is allowed to be. It uses sine and cosine terms to model the seasonal cycle. A higher value means that seasonal patterns can be flexible and detailed, while a lower values means smoother and simpler seasonality. Low values are between 3-5, medium values range from 10-15, and high values are over 20.





# Predicting



