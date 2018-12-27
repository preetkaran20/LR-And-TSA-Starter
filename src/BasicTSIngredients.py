# @Author KSASAN I317174

"""
Basic Learning for TimeSeries and its Ingredients
"""

import matplotlib.pyplot as plt
import pandas
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import stattools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

if __name__ == '__main__':
    print "Learning Basics of TSA"

    # 1) Reading CSV using Pandas.
    # 2) Making date column as Index Column.
    # 3) Paring Date Column as Date DataType
    # 4) Used Kaggle Data (not properly. Just for understanding)
    # 5) Ignored the other factors and only considered Sales.

    dataFrame = pandas.read_csv("../resources/KaggleDataForTimeSeries_TrainingDataSet.csv", parse_dates=['date'],
                                index_col='date')

    print dataFrame.head(5)  # (For getting first 5 Rows from DataSet)
    # lag Function
    # print dataFrame.shift()  # to do lag. for 1st day it will give NaN

    # Making DataFrame as TimeSeries
    timeSeries = dataFrame['sales']

    # Plot of TimeSeries :- shifting by 1 or Lag is 1.
    # 2 plots showing diff is doing same as difference between TimeSeries and its shift
    plt.plot(timeSeries.head(100) - timeSeries.shift().head(100) + 20, color='red')
    plt.plot(timeSeries.head(100).diff(), color='black')
    plt.show()

    # Simple plot with and without shift (Without and with lag)
    plt.plot(timeSeries.head(100), color='red')
    plt.plot(timeSeries.head(100).shift(1), color='black')
    plt.show()

    # Rolling window slowly response to changes (moving average) or smoothing
    # Expanding Window cumulative sum. info from long time.
    rolling = timeSeries.rolling(window=100)
    timeSeries.head(1000).plot(color='black', kind='line')
    rolling.mean().head(1000).plot(color='red', kind='line')
    # rolling.min()['sales'].head(1000).plot(color='red', kind='line')
    plt.show()

    """
    you can make your window size smaller or larger.
    Smoothing is done to reduce the noise.
    
    Window size is as per Domain Knowledge
    """

    """
    Confidence intervals are drawn as a cone. By default, this is set to a 95% confidence interval, 
    suggesting that correlation values outside of this code are very likely a correlation and not a statistical fluke.
    """
    plot_acf(timeSeries.head(2000))
    plt.show()

    plot_pacf(dataFrame.head(2000))
    plt.show()

    timeSeries = timeSeries.diff(1).dropna()
    acfResult = stattools.acf(timeSeries.head(8000), nlags=500)
    plt.plot(acfResult)
    plt.show()

    adFullerTest = adfuller(timeSeries.head(8000))
    print('ADF Statistic: %f' % adFullerTest[0])
    print('p-value: %f' % adFullerTest[1])
    print('Critical Values:')
    for key, value in adFullerTest[4].items():
        print('\t%s: %.3f' % (key, value))

    """
    Interpreting Dickey Fuller test :- 
    1) The more negative ADF statistic, the more likely we are to reject the null hypothesis 
       ie (we have a stationary dataset).
    2) As part of the output, we get a look-up table to help determine the ADF statistic. 
       We can see that our ADF statistic value is less than the value of 1%.
       This suggests that we can reject the null hypothesis with a significance level of less than 1% 
       (i.e. a low probability that the result is a statistical fluke). (more than 99% stationary TimeSeries)
    3) p-value > 0.05: Accept the null hypothesis (H0), the data has a unit root and is non-stationary.
       p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

    """

    autocorrelation_plot(timeSeries.head(8000))
    plt.show()

    lag_plot(timeSeries.head(8000))
    plt.show()

    arima = ARIMA(timeSeries.head(1005), order=(1, 1, 0))
    arima_result = arima.fit()
    print sum((arima_result.fittedvalues.head(1000).values - timeSeries.head(1000)) ** 2) / 1000
    plt.plot(timeSeries.head(1000), color='red')
    plt.plot(arima_result.fittedvalues.head(1000), color='black')
    plt.show()

    result = seasonal_decompose(timeSeries.head(366), model='additive')
    result.plot()
    plt.show()

    sarima = SARIMAX(timeSeries.head(1005), order=(1, 1, 0), seasonal_order=(1, 1, 0, 7))
    sarima_result = arima.fit()
    print sum((sarima_result.fittedvalues.head(1000).values - timeSeries.head(1000)) ** 2) / 1000
    plt.plot(timeSeries.head(1000), color='red')
    plt.plot(sarima_result.fittedvalues.head(1000), color='black')
    plt.show()
