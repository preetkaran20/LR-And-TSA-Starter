# @Author KSASAN I317174

"""
DBLoad is Trending (Either downward or upward) plus it has Seasonality too
so we are going with TimeSeries Forecasting model

Plan for Getting DBLoad is from Zabbix and we will be running a Job to get the data and save in Our tables

Comparison of ETS Vs Halt's Winter Method :-
ETS is only for Constant Modeling ie there should not be any Trend or Seasonality
Holt's Winter method is an Extension to ETS model with Inclusion of Slope and it handles both the cases mentioned above
"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing


class CreateHoltWinterModel:

    def __init__(self, dataSet):
        self.dataSet = dataSet

    def createModel(self, trend, seasonal, seasonal_periods):
        holtWinter = ExponentialSmoothing(self.dataSet, trend=trend, damped=False, seasonal=seasonal,
                                          seasonal_periods=seasonal_periods, freq='D')
        return holtWinter

    def process(self, trend, seasonal, seasonal_periods, smoothing_level, smoothing_slope, smoothing_seasonal):
        model = self.createModel(trend, seasonal, seasonal_periods)
        holtWintersResults = model.fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope,
                                       smoothing_seasonal=smoothing_seasonal)
        return holtWintersResults
