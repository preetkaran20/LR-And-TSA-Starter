# @Author KSASAN I317174

"""
We already know that it will have Error Term relation with Size and
here Linear Regression assumptions might also not fulfill

so the way is to have categories and then in each category take maximum error percentage
and then if there are many entries then take max 10 entries and do average and
this is the maximum error average and when we predict
we will add this error percentage and that will be the result
"""
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class NewRegressionApproach:

    def __init__(self, dataSet, columnIndexForClassification, maxIndex):
        self.dataSet = dataSet
        self.dataSet = self.dataSet[self.dataSet[:, columnIndexForClassification].argsort()]
        dataSetArray = []
        # Dividing into 5 groups of 50
        dataSetOfMaxIndex = None
        for index, value in enumerate(self.dataSet):
            if index % maxIndex == 0:
                if index != 0:
                    dataSetArray.append(dataSetOfMaxIndex)
                dataSetOfMaxIndex = numpy.empty([1, self.dataSet.shape[1]])
                dataSetOfMaxIndex[0] = value
            else:
                dataSetOfMaxIndex = numpy.vstack([dataSetOfMaxIndex, value])

        if dataSetOfMaxIndex.any():
            dataSetArray.append(dataSetOfMaxIndex)

        for index, dataSetEntry in enumerate(dataSetArray):
            print "\n\nEvaluating for Index :- " + str(index)
            self.evaluate(dataSetEntry)

    def evaluate(self, dataSet):
        model = Ridge(alpha=0.01, normalize=True)
        kfold = KFold(n_splits=5, random_state=12, shuffle=True)
        errorArrayTraining = numpy.empty([1,  self.dataSet.shape[1]])
        errorArrayTesting = numpy.empty([1,  self.dataSet.shape[1]])
        for trainIndex, testIndex in kfold.split(dataSet):
            inputTrainingSet = self.dataSet[:, 2:][trainIndex]
            outputTrainingSet = self.dataSet[:, 1:2][trainIndex].dot(1.0)

            inputTestingSet = self.dataSet[:, 2:][testIndex]
            outputTestingSet = self.dataSet[:, 1:2][testIndex].dot(1.0)
            model.fit(inputTrainingSet, outputTrainingSet)
            testPredictedSet = model.predict(inputTestingSet)
            trainPredictedSet = model.predict(inputTrainingSet)
            errorArrayTraining = numpy.column_stack(((outputTrainingSet - trainPredictedSet) ** 2, outputTrainingSet,
                                                     trainPredictedSet,
                                                     numpy.abs(
                                                         trainPredictedSet - outputTrainingSet) * 100 / outputTrainingSet))
            errorArrayTesting = numpy.column_stack(((outputTestingSet - testPredictedSet) ** 2, outputTestingSet,
                                                    testPredictedSet,
                                                    numpy.abs(
                                                        testPredictedSet - outputTestingSet) * 100 / outputTestingSet))

            print sum(errorArrayTraining[errorArrayTraining[:, 3].argsort()][-10:][:, 3]) / 10
            print sum(errorArrayTesting[errorArrayTesting[:, 3].argsort()][-10:][:, 3]) / 10
