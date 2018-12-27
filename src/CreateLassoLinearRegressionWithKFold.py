# @Author KSASAN I317174
import numpy
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

import LinearRegressionUtility


class CreateLassoLinearRegressionWithKFold:

    def __init__(self, dataSet):
        self.dataSet = dataSet

    def preProcessTrainingSet(self, degree, inputTrainingSet, printParameters=False):
        """
        Polynomial Regression
        :param degree:
        :return:
        """
        polynomial_Feature = PolynomialFeatures(degree=degree)
        inputTrainingSet_poly = polynomial_Feature.fit_transform(inputTrainingSet)
        if printParameters:
            print str(
                polynomial_Feature.get_feature_names(['Disk Size', 'Row Store Size', 'Column Store Size', 'Delta']))
        return inputTrainingSet_poly

    def __getModel(self, printModel=False):
        # type: () -> Lasso
        lasso = Lasso(max_iter=100000, alpha=0.01, normalize=True, copy_X=True)
        if printModel:
            print "\n\n *********************** Lasso Regression ********************* \n"
            print " Factors :- \n" + str(lasso)
        return lasso

    def __getKFold(self, numberOfFolds):
        # type: (numberOfFolds) -> int
        # type: () -> KFold
        return KFold(n_splits=numberOfFolds, random_state=12, shuffle=True)

    def prepareModel(self, inputTrainingSet, outputTrainingSet, printModel=False):
        """
        PreparedModel and returns the Model based on TrainingIndex
        :param trainIndex:
        :return:
        """
        lasso = self.__getModel(printModel)
        lasso.fit(inputTrainingSet, outputTrainingSet)
        return lasso

    def evaluateModel(self, jTrainError, jTestError, coefficient):
        print "Training Error : " + str(jTrainError)
        print "Testing Error : " + str(jTestError)
        print "Coefficients : " + str(coefficient)


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
        for trainIndex, testIndex in kfold.split(self.dataSet):
            inputTrainingSet = self.dataSet[:, 2:][trainIndex]
            outputTrainingSet = self.dataSet[:, 1:2][trainIndex].dot(1.0)

            inputTestingSet = self.dataSet[:, 2:][testIndex]
            outputTestingSet = self.dataSet[:, 1:2][testIndex].dot(1.0)

            inputTrainingSet = self.preProcessTrainingSet(2, inputTrainingSet, isFirstTime)
            inputTestingSet = self.preProcessTrainingSet(2, inputTestingSet)

            model = self.prepareModel(inputTrainingSet, outputTrainingSet, isFirstTime)
            # Reshape is done as Lasso returns Array as [] instead of [[],[]] which is not proper for Residual Plot
            testPredictedSet = model.predict(inputTestingSet).reshape(outputTestingSet.size, 1)
            trainPredictedSet = model.predict(inputTrainingSet).reshape(outputTrainingSet.size, 1)

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
                           (sumMeanSquaredErrorInTest / numberOfTimeCalled), coefficients)

        LinearRegressionUtility.createGraphs(allOutputTrainingSet, allOutputPredictedTrainingSet, allOutputTestingSet,
                                             allOutputPredictedTestingSet)
        LinearRegressionUtility.createLearningCurve(self.__getModel(),
                                                    self.preProcessTrainingSet(2, self.dataSet[:, 2:]),
                                                   self.dataSet[:, 1:2], self.__getKFold(numberOfFolds))
        LinearRegressionUtility.createValidationCurvAsParam(Lasso(normalize=True), "alpha",
                                                            self.preProcessTrainingSet(2,
                                                                                      self.dataSet[:, 2:]),
                                                           self.dataSet[:, 1:2],
                                                            self.__getKFold(numberOfFolds))

        print "\n*******************************Lasso Ends*******************************\n"
