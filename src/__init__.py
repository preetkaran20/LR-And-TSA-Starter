# @Author KSASAN I317174

"""
This is the Entry point for all the ML Related Stuff
"""
from plotly.utils import numpy
from sklearn.preprocessing import LabelEncoder

import LinearRegressionUtility
from CreateLassoLinearRegressionWithKFold import CreateLassoLinearRegressionWithKFold
from CreateLinearRegressionModelWithKFold import CreateLinearRegressionModelWithKFold
from CreateRidgeLinearRegressionWithKFold import CreateRidgeLinearRegressionWithKFold
from NewRegressionApproach import NewRegressionApproach

if __name__ == '__main__':
    # Creating Graphs for Export Hana
    exportDataSet = LinearRegressionUtility.loadDataSet("../resources/DBPoolDataExport.csv", ",", False)
    LinearRegressionUtility.createMLModelAndGraphs(exportDataSet, 5)

    # Creating Graphs for Import Hana
    importDataSet = LinearRegressionUtility.loadDataSetOfDiffDataTypesAsString(
        "../resources/DBPoolDataImport.csv", ",", False)
    labelEncoder = LabelEncoder()
    importDataSet[:, 3] = labelEncoder.fit_transform(importDataSet[:, 3].astype('str'))
    importDataSet = importDataSet.astype(numpy.float64)

    # LinearRegressionUtility.removeColumns(importDataSet, 3)
    # LinearRegressionUtility.createMLModelAndGraphs(importDataSet, 5)

    # Classify and Regress Approach as Discussed with @Roopang
    importDataSet = LinearRegressionUtility.removeColumns(importDataSet, 3)
    newRegressionApproach = NewRegressionApproach(importDataSet, 2, 50)
    newRegressionApproach = NewRegressionApproach(exportDataSet, 2, 50)