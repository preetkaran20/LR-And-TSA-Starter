# @Author KSASAN I317174
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
import LinearRegressionUtility


class CreateLinearRegressionModelWithKFold:

    def __init__(self, dataSet):
        self.dataSet = dataSet

    def __getModel(self, printModel=False):
        # type: () -> LinearRegression
        linearRegression = LinearRegression()
        if printModel:
            print "\n\n *********************** Linear Regression ********************* \n"
            print " Factors :-  \n" + str(linearRegression)
        return linearRegression

    def __getKFold(self, numberOfFolds):
        # type: (numberOfFolds) -> int
        # type: () -> KFold
        return KFold(n_splits=numberOfFolds, random_state=12, shuffle=True)

    def prepareModel(self, inputTrainingSet, outputTrainingSet, printModel):
        """
        PreparedModel and returns the Model based on TrainingIndex
        :param trainIndex:
        :return:
        """
        linearRegression = self.__getModel(printModel)
        linearRegression.fit(inputTrainingSet, outputTrainingSet)
        return linearRegression

    def evaluateModel(self, jTrainError, jTestError, coefficient):
        print "Training Error : " + str(jTrainError)
        print "Testing Error : " + str(jTestError)
        print "Coefficients : " + str(coefficient)

    def preProcessTrainingSet(self, degree, inputTrainingSet):
        """
        Polynomial Regression
        :param degree:
        :return:
        """
        polynomial_Feature = PolynomialFeatures(degree=degree)
        inputTrainingSet_poly = polynomial_Feature.fit_transform(inputTrainingSet)
        return inputTrainingSet_poly

    def __preparingForEvaluation__(self, outputTestingSet, outputTrainingSet, trainPredictedSet, testPredictedSet, model,
                                 isFirstTime):
        """
        :param outputTestingSet:
        :param outputTrainingSet:
        :param trainPredictedSet:
        :param testPredictedSet:
        :param model:
        :param isFirstTime:
        """
        global allOutputTrainingSet
        global allOutputPredictedTrainingSet
        global allOutputTestingSet
        global allOutputPredictedTestingSet
        global coefficients

        if (isFirstTime):
            allOutputTestingSet = outputTestingSet
            allOutputTrainingSet = outputTrainingSet
            allOutputPredictedTrainingSet = trainPredictedSet
            allOutputPredictedTestingSet = testPredictedSet
            coefficients = model.coef_
        else:
            allOutputPredictedTestingSet = numpy.append(allOutputPredictedTestingSet, testPredictedSet)
            allOutputPredictedTrainingSet = numpy.append(allOutputPredictedTrainingSet, trainPredictedSet)
            allOutputTestingSet = numpy.append(allOutputTestingSet, outputTestingSet)
            allOutputTrainingSet = numpy.append(allOutputTrainingSet, outputTrainingSet)
            coefficients = numpy.add(coefficients, model.coef_)

    def process(self, numberOfFolds):
        """
        :param inputTrainingSet:
        :param outputTrainingSet:
        :param numberOfFolds:
        """
        sumMeanSquaredErrorInTest = 0.0
        sumMeanSquaredErrorInTrain = 0.0
        numberOfTimeCalled = 0

        global allOutputTrainingSet
        global allOutputPredictedTrainingSet
        global allOutputTestingSet
        global allOutputPredictedTestingSet
        global coefficients

        isFirstTime = True

        kfold = self.__getKFold(numberOfFolds)
        #self.dataSet = self.dataSet[self.dataSet[:, 1].argsort()]
        for trainIndex, testIndex in kfold.split(self.dataSet):
            inputTrainingSet = self.dataSet[:, 2:][trainIndex]
            outputTrainingSet = self.dataSet[:, 1:2][trainIndex].dot(1.0)

            inputTestingSet = self.dataSet[:, 2:][testIndex]
            outputTestingSet = self.dataSet[:, 1:2][testIndex].dot(1.0)

            inputTrainingSet = self.preProcessTrainingSet(1, inputTrainingSet)
            inputTestingSet = self.preProcessTrainingSet(1, inputTestingSet)

            model = self.prepareModel(inputTrainingSet, outputTrainingSet, isFirstTime)
            testPredictedSet = model.predict(inputTestingSet)
            trainPredictedSet = model.predict(inputTrainingSet)
            print(testPredictedSet)
            print(trainPredictedSet)
            if (isFirstTime):
                self.__preparingForEvaluation__(outputTestingSet, outputTrainingSet, trainPredictedSet, testPredictedSet,
                                              model,
                                              isFirstTime)
                isFirstTime = False
            else:
                self.__preparingForEvaluation__(outputTestingSet, outputTrainingSet, trainPredictedSet, testPredictedSet,
                                              model,
                                              isFirstTime)

            # Finding Jcv and Jtrain
            sumMeanSquaredErrorInTest = sumMeanSquaredErrorInTest + mean_squared_error(outputTestingSet,
                                                                                       testPredictedSet)
            sumMeanSquaredErrorInTrain = sumMeanSquaredErrorInTrain + mean_squared_error(
                outputTrainingSet,
                trainPredictedSet)

            print mean_squared_error(outputTestingSet, testPredictedSet)
            print mean_squared_error(outputTrainingSet, trainPredictedSet)
            print model.coef_

            numberOfTimeCalled = numberOfTimeCalled + 1

        self.evaluateModel((sumMeanSquaredErrorInTrain / numberOfTimeCalled),
                           (sumMeanSquaredErrorInTest / numberOfTimeCalled), (coefficients / numberOfTimeCalled))

        LinearRegressionUtility.createGraphs(allOutputTrainingSet, allOutputPredictedTrainingSet, allOutputTestingSet,
                                             allOutputPredictedTestingSet)
        LinearRegressionUtility.createLearningCurve(self.__getModel(),
                                                   self.dataSet[:, 2:],
                                                   self.dataSet[:, 1:2],
                                                    self.__getKFold(numberOfFolds))
        print "\n*******************************Linear Regression Ends*******************************\n"
