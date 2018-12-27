#
import json

import matplotlib.pyplot as plt
import pandas
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def drawRollingMeanAndStdWithTimeSeries(timeSeries, windowSize=100):
    """
    Used for Drawing Rolling Mean and Rolling Standard Deviation with Respect to TimeSeries
    :param timeSeries: Pandas Series
    :param windowSize: int
    """
    rolmean = timeSeries.rolling(window=windowSize).mean()
    rolstd = timeSeries.rolling(window=windowSize).std()
    timeSeries.plot(color='blue', label='Original')
    # rolmean.plot(color='red', label='Rolling Mean',  marker='.', linewidth=0)
    # rolstd.plot(color='black', label='Rolling Std', marker='.', linewidth=0)
    rolmean.plot(color='red', label='Rolling Mean')
    rolstd.plot(color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.show()


def computeADFullerTest(timeSeries):
    """
    Computed Adjusted Dickey Fuller Test for TimeSeries and prints it
    :param timeSeries:Pandas TimeSeries
    """
    adFullerTest = adfuller(timeSeries)
    print('ADF Statistic: %f' % adFullerTest[0])
    print('p-value: %f' % adFullerTest[1])
    print('Critical Values:')
    for key, value in adFullerTest[4].items():
        print('\t%s: %.3f' % (key, value))


def getCumSumForIntervalData(actualTimeSeries, predictedValuesOnLaggedData, intervalSize=365):
    """
    :param actualTimeSeries: Object of DataFrame TimeSeries
    :param predictedValuesOnLaggedData: Object of DataFrame TimeSeries
    :param intervalSize: Period or Seasonality
    :return: Cumulative Sum for given Interval (dtype will be Float please consider properly)
    """
    arrayOfSeasonal = pandas.Series(actualTimeSeries[:intervalSize], copy=True)
    for index, value in enumerate(predictedValuesOnLaggedData):
        predictedValuesOnLaggedData[index] = value + arrayOfSeasonal[index % intervalSize]
        arrayOfSeasonal[index % intervalSize] = predictedValuesOnLaggedData[index]
    return pandas.Series(actualTimeSeries[:intervalSize], actualTimeSeries.index, copy=True).add(
        predictedValuesOnLaggedData,
        fill_value=0)


def getKaggleTrainingData():
    """
    :return: TimeSeries by reading KaggleData
    """
    dataFrame = pandas.read_csv("../resources/KaggleDataForTimeSeries_TrainingDataSet.csv", parse_dates=['date'],
                                index_col='date')
    # Note:- Only Considered one Store and one Item
    timeSeries = dataFrame['sales'].head(1826)
    return timeSeries


def createAutoCorrelationAndPartialCorrelationGraph(timeseries_seasonal_lag):
    """
    used to create ACF and PACF plots for TimeSeries
    :param timeseries_seasonal_lag:
    """
    plot_acf(timeseries_seasonal_lag)
    plt.show()

    plot_pacf(timeseries_seasonal_lag.head(500), method='ols')
    plt.show()


def parseJsonData():
    """
    Parses that Json file and returns that as a DataStructure
    :return:
    """
    with open('../resources/SimpleDCProductionData_Automated.json') as json_data:
        jsonPyDS = json.load(json_data)
        json_data.close()

    return jsonPyDS


def convertOdataJsonToPandasTS(jsonPyDS, logging=False):
    """
    Converting Odata Response from https://itoahana.sapsf.com/sfsf/hdbmetrics/itoametrics_reapp.xsodata/kpi?$format=json
    to Pandas TimeSeries for more analysis
    :param jsonPyDS:
    :param logging:
    :return:
    """
    columnsToBeRemovedFromJson = ['DC', 'IDA', 'PRODUCT', 'ROLE', 'HOST', 'DATABASE_NAME', 'UP_SINCE', '__metadata']
    listOfJsonObjects = jsonPyDS
    if logging:
        for jsonObject in listOfJsonObjects:
            print jsonObject

    dataFrame = pandas.DataFrame(listOfJsonObjects)
    dataFrame['LAST_CONNECTED'].replace('/Date\(', '', regex=True, inplace=True)
    dataFrame['LAST_CONNECTED'].replace(
        '\)/', '', regex=True, inplace=True)
    dataFrame['LAST_CONNECTED'] = pandas.to_datetime(pandas.Series(dataFrame['LAST_CONNECTED']), unit='ms')

    if logging:
        print dataFrame['LAST_CONNECTED']
    dataFrame.set_index('LAST_CONNECTED', inplace=True)
    print dataFrame.sort_index().index.unique()
    dataFrame.sort_index(inplace=True)
    # Only concentrating on one Host out of all the DataFrame.
    # As we are going to predict for one host
    # As iterations are slow we are going with the methods present in Pandas
    newDataFrame = dataFrame[dataFrame['HOST'] == 'pc12hdbbc802']
    newDataFrame = newDataFrame.drop(columnsToBeRemovedFromJson, axis=1)
    newDataFrame = newDataFrame.apply(pandas.to_numeric)
    newDataFrame.plot(label='Original')
    plt.legend(loc='best')
    plt.show()
