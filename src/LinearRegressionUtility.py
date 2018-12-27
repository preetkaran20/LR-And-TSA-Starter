# @Author KSASAN I317174

import numpy
from matplotlib import pyplot as plt
import pylab
import statsmodels.api as stats
from sklearn.model_selection import learning_curve, KFold, validation_curve
from CreateLinearRegressionModelWithKFold import CreateLinearRegressionModelWithKFold
from CreateRidgeLinearRegressionWithKFold import CreateRidgeLinearRegressionWithKFold
from CreateLassoLinearRegressionWithKFold import CreateLassoLinearRegressionWithKFold


def loadDataSet(fileName, delimiter, isReshapeNeeded):
    """
    Loads data from param path provided using Numpy and returns Numpy Array
    We may need to shape the data properly if not a 2D array
    :param fileName: String
    :param delimiter: ', or ;'
    :param isReshapeNeeded: only needed if input is having only 1D array
    :return:
    """
    dataSet = numpy.loadtxt(fname=fileName, delimiter=delimiter)
    if isReshapeNeeded:
        dataSet.reshape(len(dataSet), 1)
    return dataSet


def loadDataSetOfDiffDataTypesAsString(fileName, delimiter, isReshapeNeeded):
    """
    Load Training Set in case Training Set contains String and Int.
    Loads all the data as String then it can be converted to Categorical Data
    using LabelEncoder.Hence converting all the data as Float or Int
    :param fileName:
    :param delimiter:
    :param isReshapeNeeded:
    :return:
    """
    dataSet = numpy.genfromtxt(fname=fileName, dtype='|S', delimiter=delimiter)
    if isReshapeNeeded:
        dataSet.reshape(len(dataSet), 1)
    return dataSet


def removeColumns(inputTrainingSet, columnNumber):
    """
    :param inputTrainingSet: Numpy Array
    :param columnNumber: Integer starts with 0
    :return:
    """
    inputTrainingSet = numpy.delete(inputTrainingSet, columnNumber, 1)
    return inputTrainingSet


def __createResidualAndQQPlots__(allOutputTrainingSet, allOutputPredictedTrainingSet, allOutputTestingSet,
                                 allOutputPredictedTestingSet):
    """
    Creates Residual Plot and Q-Q plots which will help in model evaluation
    For both TrainingSet and Testing Set
    :param allOutputTrainingSet:
    :param allOutputPredictedTrainingSet:
    :param allOutputTestingSet:
    :param allOutputPredictedTestingSet:
    """
    plt.title("Residual Plot of Testing Set")
    plt.xlabel("Predicted Value Testing")
    plt.ylabel("Residual")
    residualTestingSet = numpy.subtract(allOutputTestingSet, allOutputPredictedTestingSet)
    plt.scatter(allOutputPredictedTestingSet, residualTestingSet)
    plt.show()

    plt.title("Residual Plot of Training Set")
    plt.xlabel("Predicted Value Training")
    plt.ylabel("Residual")
    residualTrainingSet = numpy.subtract(allOutputTrainingSet, allOutputPredictedTrainingSet)
    plt.scatter(allOutputPredictedTrainingSet, residualTrainingSet)
    plt.show()

    stats.qqplot(residualTrainingSet, dist="norm", line='s')
    pylab.show()

    stats.qqplot(residualTestingSet, dist="norm", line='s')
    pylab.show()


def createGraphs(allOutputTrainingSet, allOutputPredictedTrainingSet, allOutputTestingSet,
                 allOutputPredictedTestingSet):
    """
    Creates Graphs which will help in Evaluating the Model

    :param allOutputTrainingSet: Numpy Array
    :param allOutputPredictedTrainingSet: Numpy Array
    :param allOutputTestingSet: Numpy Array
    :param allOutputPredictedTestingSet: Numpy Array
    """

    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("Actual Value Vs Predicted Value")
    plt.scatter(allOutputTrainingSet, allOutputPredictedTrainingSet, color='g')
    plt.scatter(allOutputTestingSet, allOutputPredictedTestingSet, color='g')
    plt.plot([0, 1000], [0, 1000], "k-")
    plt.show()

    plt.xlabel("Index")
    plt.ylabel("Actual Value or Predicted Value")
    plt.title("Index Graph")
    plt.scatter(numpy.arange(allOutputPredictedTrainingSet.size), allOutputPredictedTrainingSet, color='g', s=2,
                label="Predicted Value")
    plt.scatter(numpy.arange(allOutputTrainingSet.size), allOutputTrainingSet, color='r', s=2, label="Actual Value")
    for i in range(allOutputPredictedTrainingSet.size):
        plt.plot([i, i], [allOutputPredictedTrainingSet[i], allOutputTrainingSet[i]], "k-")
    plt.scatter(numpy.arange(allOutputPredictedTestingSet.size), allOutputPredictedTestingSet, color='g', s=2)
    plt.scatter(numpy.arange(allOutputTestingSet.size), allOutputTestingSet, color='r', s=2)
    for i in range(allOutputTestingSet.size):
        plt.plot([i, i], [allOutputPredictedTestingSet[i], allOutputTestingSet[i]], "k-")
    # plt.plot([0, 1000], [0, 1000], "k--")
    plt.legend(loc="best")
    plt.show()

    plt.xlabel("Index")
    plt.ylabel("Actual Value or Predicted Value")
    plt.title("Index Graph")
    plt.scatter(numpy.arange(allOutputPredictedTestingSet.size), allOutputPredictedTestingSet, color='g', s=2,
                label="Predicted Value")
    plt.scatter(numpy.arange(allOutputTestingSet.size), allOutputTestingSet, color='r', s=2, label="Actual Value")
    plt.scatter(numpy.arange(allOutputPredictedTrainingSet.size), allOutputPredictedTrainingSet, color='g', s=2)
    plt.scatter(numpy.arange(allOutputTrainingSet.size), allOutputTrainingSet, color='r', s=2)
    plt.legend(loc="best")
    plt.show()

    __createResidualAndQQPlots__(allOutputTrainingSet, allOutputPredictedTrainingSet, allOutputTestingSet,
                                 allOutputPredictedTestingSet)


def createLearningCurve(clf, inputSet, outputSet, kfold):
    """
    This is used to create Learning Curve.

    Possible values for Scorer or Scoring are :-
    ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision',
    'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
    'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'neg_log_loss',
    'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
    'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
    'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
    'recall_weighted', 'roc_auc', 'v_measure_score']

    train_size is how many chunks to divide the DataSet

    :param kfold: K-Fold
    :param clf: Model
    :param inputSet: Numpy Array
    :param outputSet: Numpy Array
    """
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, inputSet, outputSet, cv=kfold,
        scoring="neg_mean_squared_error", n_jobs=1,
        train_sizes=numpy.linspace(0.1, 1.0, 20), shuffle=True)
    train_scores = -1.0 * train_scores
    test_scores = -1.0 * test_scores
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def __createValidationCurv__(clf, param_name, plt, inputSet, outputSet, kfold):
    """
    :param clf: Model
    :param param_name: estimator.get_params().keys()
    :param plt: Polyplot
    :param inputSet: Numpy Array
    :param outputSet: Numpy Array
    :param kfold: K-Fold
    """
    # type: (Model, Boolean, polyplt, inputSet, outputSet, KFold) -> void
    param_range = numpy.linspace(0.0001, 1, 1000)
    train_scores, test_scores = validation_curve(
        clf, inputSet, outputSet, param_name, param_range=param_range,
        cv=kfold, scoring="neg_mean_squared_error", n_jobs=1)

    train_scores = -1.0 * train_scores
    test_scores = -1.0 * test_scores
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)

    lw = 2
    plt.ylim(-100, 1200)
    plt.grid()
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


def createValidationCurvAsParam(clf, param_name, inputSet, outputSet, kfold):
    """
    This will create a Curve which is between J(@) and ParamName
    It will give an indication of values of Lambda and Degree which will give better result.

    in case of NonRegularized Classifier we are only concerned with Degree but in case of
    Regularized Classifier we are concerned with both Degree and Lambda which is Regularized parameter

    :param kfold: K-FOLD
    :param inputSet: Numpy Array
    :param outputSet: Numpy Array
    :param param_name: Values of param_name can be found by running "estimator.get_params().keys()"
                        Estimator is Classifier or Model
    :param clf: Classifier/Model
    """

    plt.figure()
    plt.title("Validation Curve")
    plt.ylabel("Mean Squared Error Score")
    plt.grid()
    __createValidationCurv__(clf, param_name, plt, inputSet, outputSet, kfold)
    plt.xlabel(param_name)


def createMLModelAndGraphs(dataSet, folds):
    """
    Create all present basic models in CodeBase with the DataSet

    Namely :-
    1) createLinearRegressionModelWithKFold
    2) createRidgeLinearRegressionWithKFold
    3) createLassoLinearRegressionWithKFold

    :param dataSet: Numpy Array
    :param folds: int for KFold
    """
    createLinearRegressionModelWithKFold = CreateLinearRegressionModelWithKFold(dataSet)
    createLinearRegressionModelWithKFold.process(folds)

    createRidgeLinearRegressionWithKFold = CreateRidgeLinearRegressionWithKFold(dataSet)
    createRidgeLinearRegressionWithKFold.process(folds)

    createLassoLinearRegressionWithKFold = CreateLassoLinearRegressionWithKFold(dataSet)
    createLassoLinearRegressionWithKFold.process(folds)
