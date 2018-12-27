# @Author KSASAN I317174

"""
DBLoad in Successfactors is Trending (Either downward or upward) plus it has Seasonality too
so we are going with TimeSeries Forecasting model

Plan for Getting DBLoad is from ITOA and we will be running a Job to get the data and save in Our tables

ARIMA is AutoRegressive Integrated Moving Average
There are other Statistical Models are also there like ETS (Estimated Time Smoothing) and Halt's Winter Method

Major Issues with ARIMA is it needs Stationarity in DataSet (ie Constant Mean and Variance)
which is very difficult to have in Real Case Scenarios.

So there are some ways to Make Model as Stationarity and then Process and move forward.

Please Provide Stationary TimeSeries as this is Basic Assumption
"""
from pandas import infer_freq
from statsmodels.tsa.arima_model import ARIMA


class CreateStatsARIMAModel:

    def __init__(self, timeSeries, order):
        """
        :param timeSeries:pandas TimeSeries
        :param order: (p,d,q) {PACF value/Differencing(if Needed)/ACF Value}
        """
        self.timeSeries = timeSeries
        self.order = order

    def createModel(self):
        """
        used to create ARIMA Model
        :return:
        """
        model = ARIMA(self.timeSeries, order=self.order, freq=infer_freq(self.timeSeries.index))
        return model

    def evaluteModel(self, modelResult):
        """
        used to evalute Model
        :param modelResult:
        """
        print sum((modelResult.fittedvalues.values - self.timeSeries) ** 2) / len(self.timeSeries)

    def process(self):
        model = self.createModel()
        result = model.fit(disp=0)
        self.evaluteModel(result)
        return result.fittedvalues
