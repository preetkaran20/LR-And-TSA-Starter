# Author I317174 KSASAN

import unittest

import src.TimeSeriesAnalysisUtility


class TestingTSUtility(unittest.TestCase):
    def test_CumSumForIntervalData(self):
        """
        Used to Test TSUtility's Cumsum for Interval Data
        """
        timeSeries = src.TimeSeriesAnalysisUtility.getKaggleTrainingData()
        timeseries_diff_lag100 = src.TimeSeriesAnalysisUtility.getCumSumForIntervalData(timeSeries,
                                                                                        timeSeries.diff(
                                                                                        100).dropna().astype(int), 100)
        self.assertTrue(timeSeries.astype(float).equals(timeseries_diff_lag100), "Series are not equal")


if __name__ == '__main__':
    unittest.main()
