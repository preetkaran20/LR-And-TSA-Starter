# @Author KSASAN I317174

"""
Making Recipe from the Ingredients from BasicTSIngredients
"""
import matplotlib.pyplot as plt
import pandas
from statsmodels.tsa.seasonal import seasonal_decompose

import TimeSeriesAnalysisUtility
from CreateStatsARIMAModel import CreateStatsARIMAModel
from CreateHoltWinterModel import CreateHoltWinterModel


def handleARIMASeasonalDecompositionForecast(timeSeries, seasonal_lag):
    """
    Used to give an understanding on How to use Seasonal Decomposition for TimeSeries Forecast
    :param timeSeries:
    """
    seasonal_decompose_result = seasonal_decompose(timeSeries, freq=seasonal_lag)
    seasonal_decompose_result.plot()
    plt.show()

    print "ADFuller for Seasonal Decomposition Forecasting"
    TimeSeriesAnalysisUtility.computeADFullerTest(seasonal_decompose_result.resid.dropna())

    timeseries_diff_SeasonalLag = seasonal_decompose_result.resid.dropna()
    TimeSeriesAnalysisUtility.createAutoCorrelationAndPartialCorrelationGraph(timeseries_diff_SeasonalLag)

    arimaModel = CreateStatsARIMAModel(timeseries_diff_SeasonalLag, order=(7, 0, 0))
    fittedValues = arimaModel.process()
    plt.plot(timeseries_diff_SeasonalLag, color='red')
    plt.plot(fittedValues, color='black')
    plt.show()

    timeseries_diff_cumsum = fittedValues + seasonal_decompose_result.trend + seasonal_decompose_result.seasonal
    plt.plot(timeSeries, color='red')
    plt.plot(timeseries_diff_cumsum, color='black')
    plt.show()

    print "Error term for Seasonal decomposition" + str(
        sum((timeseries_diff_cumsum['2013-07-02':'2017-07-02'] - timeSeries['2013-07-02':'2017-07-02']) ** 2) / len(
            timeSeries['2013-07-02':'2017-07-02']))


def handleArimaSeasonalDifferencingForecast(timeSeries, seasonal_lag):
    """
    Used to give an understanding on How to use Seasonal Differencing for TimeSeries Forecast
    :param timeSeries:
    """
    timeseries_seasonal_lag = timeSeries - timeSeries.shift(seasonal_lag)
    timeseries_seasonal_lag.dropna(inplace=True)

    TimeSeriesAnalysisUtility.createAutoCorrelationAndPartialCorrelationGraph(timeseries_seasonal_lag)
    print "ADFuller for Differencing Forecast"
    TimeSeriesAnalysisUtility.computeADFullerTest(timeseries_seasonal_lag)

    arimaModel = CreateStatsARIMAModel(timeseries_seasonal_lag, order=(7, 0, 0))
    fittedValues = arimaModel.process()
    plt.plot(timeseries_seasonal_lag, color='red')
    plt.plot(fittedValues, color='black')
    plt.show()

    cumsum_interval_value = TimeSeriesAnalysisUtility.getCumSumForIntervalData(timeSeries, fittedValues, seasonal_lag)
    plt.plot(timeSeries, color='red')
    plt.plot(cumsum_interval_value, color='black')
    plt.show()

    print "Error term for Seasonal differencing :- " + str(
        sum((cumsum_interval_value - timeSeries) ** 2) / len(timeSeries))


def __createHoltWinterModel__(timeSeries):
    holtWinterModel = CreateHoltWinterModel(timeSeries)
    return holtWinterModel.process(trend='mul', seasonal='add', seasonal_periods=365, smoothing_level=0.2,
                                   smoothing_slope=0.1, smoothing_seasonal=0.1)


def handleTimeSeriesForecastingUsingHoldWinters(timeSeries):
    """
    This method handles Forecasting using HoltWinters Model
    :param timeSeries:
    """
    holtWintersResult = __createHoltWinterModel__(timeSeries)
    plt.plot(timeSeries, color='red')
    plt.plot(holtWintersResult.fittedvalues, color='black')
    plt.show()
    print sum((holtWintersResult.fittedvalues - timeSeries) ** 2) / len(timeSeries.index)


def initTimeSeriesAnalysis():
    """
    Stating point for TimeSeries Analysis using Kaggle Data Set
    """
    print "Time Series Analysis using Learning from BasicTSIngredients"
    timeSeries = TimeSeriesAnalysisUtility.getKaggleTrainingData()
    timeSeries.plot(color='black', label='Original')
    plt.legend(loc='best')
    plt.show()

    """
    Rolling Mean states that year wise there is some sin curve and Mean is not constant.
    Rolling Standard Deviation is almost constant 
    """
    TimeSeriesAnalysisUtility.drawRollingMeanAndStdWithTimeSeries(timeSeries, 100)
    TimeSeriesAnalysisUtility.computeADFullerTest(timeSeries)
    seasonal_lag = 365
    """
    For removing trend (if only trend is there without seasonality) we generally use Moving Average or Smoothing.
    eg of Moving Average is below explained.
    """
    """
    rolmean = timeSeries.rolling(window=100).mean()
    timeSeries_rolmean_diff = timeSeries - rolmean
    timeSeries_rolmean_diff.dropna(inplace=True)
    timeSeries_rolmean_diff.plot(color='black', label='Timeseries without rolling mean')
    plt.legend(loc='best')
    plt.show()

    TimeSeriesAnalysisUtility.drawRollingMeanAndStdWithTimeSeries(timeSeries_rolmean_diff, 183)
    TimeSeriesAnalysisUtility.computeADFullerTest(timeSeries_rolmean_diff)
    """

    """
    Seasonality Removal is done using decomposition or differencing
    First showing differencing and then decomposition
    Model value in Seasonal_Decompose is "model='multiplicative' or default is additive"
    """
    handleArimaSeasonalDifferencingForecast(timeSeries, seasonal_lag)
    handleARIMASeasonalDecompositionForecast(timeSeries, seasonal_lag)
    handleTimeSeriesForecastingUsingHoldWinters(timeSeries)


if __name__ == '__main__':
    # initTimeSeriesAnalysis()
    print ":In Kaggle"
    jsonPyDS = TimeSeriesAnalysisUtility.parseJsonData()
    pandasDF = TimeSeriesAnalysisUtility.convertOdataJsonToPandasTS(jsonPyDS, False)

