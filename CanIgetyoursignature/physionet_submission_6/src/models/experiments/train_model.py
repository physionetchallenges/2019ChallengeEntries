from definitions import *
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor, XGBClassifier
from src.data.transformers import FillMissing, RemoveExtremeValues
from src.models.optimizers import ThresholdOptimizer, apply_threshold
from src.models.evaluators import ComputeNormalizedUtility
from src.models.functions import CustomStratifiedGroupKFold
from src.features.transformers import SignaturesFromIdDataframe


def extract_results(df, cv, results):
    """
    Extracts information from results.

    Used when results contains a list of tuples of the following form:
        results = [(predictions_1, score_1), (predictions_2, score_2), ...]
    and vs contains the test indexes that the predictions and scores were made at
        cv = [(train_idxs_1, test_idxs_1), ...]
    So the return is the average score across folds, and the cv indexes are used to put the predictions back into the
    correct places in the dataframe to give the cv predictions
    """
    # Extract individually the predictions and scores from results
    predictions = [x[0] for x in results]
    scores = [x[1] for x in results]

    # Add the predictions to the correct test_idx locations and then make into a dataframe with the same index as df
    all_preds = pd.Series(index=df.index, data=0)
    for p in predictions:
        all_preds.loc[p.index] = p.values

    # Get the average score
    score = np.mean(scores)

    return all_preds, score


def _fill_mean(df, train_idx, test_idx):
    """ Performs the fill mean method if there are missing values (train mean used on test) on a numpy array """
    if np.isnan(df.values).sum() > 0:
        mean_filler = FillMissing()
        df.iloc[train_idx] = mean_filler.fit_transform(df.iloc[train_idx])
        df.iloc[test_idx] = mean_filler.transform(df.iloc[test_idx])
    return df


def _optimise_threshold(clf, X, y_binary):
    """ Optimises the threshold for the score function """
    predictions_train = clf.predict(X)
    best_threshold, train_score = ThresholdOptimizer().optimize(y_binary, predictions_train)
    return best_threshold, train_score


def _predict_threshold_and_score(clf, X, y_binary, threshold):
    """ Predicts, applies the threshold to the predictions and scores the result """
    predictions_proba = clf.predict(X)
    score = ComputeNormalizedUtility(threshold=threshold).score(y_binary, predictions_proba)
    predictions = (predictions_proba > threshold).astype(int)
    return predictions, predictions_proba, score


class TrainingLoop(object):
    """
    Performs the main loop of training for one cv fold.

    Steps:
        1. Fills in any missing values (currently uses the feature means of the training set)
        2. Computes and appends signatures
        3. Trains a classifier
        4. Optimises the threshold on the training set
        5. Evaluates on the validation set and returns the predictions and scores
    """
    def __init__(self, df, labels_utility, labels_binary):
        self.df = df
        self.labels_utility = labels_utility
        self.labels_binary = labels_binary

        # Set the clf
        self.clf = XGBRegressor()


    def prediction_loop(self, train_idx, test_idx):
        """
        Basic prediction loop over 1 cross validated fold.

        :param clf: Classifier
        :param X: inputs
        :param y: labels
        :param y_binary: Binary labels (since y is normally regressed on)
        :param train_idx: Training indexes
        :param test_idx: Testing indexes to give the score on
        :return: score, score on the test labels optimised on the valdiation
        """
        # Open up our params
        df, labels_utility, labels_binary, clf = self.df, self.labels_utility, self.labels_binary, self.clf

        # Put in machine learning form
        X = df.values
        y = labels_utility.values

        # Add params and fit the classifier to the training data
        clf.fit(X[train_idx], y[train_idx])

        # Optimise the threshold on the training set
        threshold, train_score = _optimise_threshold(clf, X[train_idx], labels_binary.iloc[train_idx])

        # Evaluate on the validation set
        predictions, pred_proba, score = _predict_threshold_and_score(clf, X[test_idx], labels_binary.iloc[test_idx],
                                                                      threshold=threshold)
        predictions = pd.Series(index=df.iloc[test_idx].index, data=predictions)
        pred_proba = pd.Series(index=df.iloc[test_idx].index, data=pred_proba)

        # Print information
        print('\tScore on cv fold: {:.3f}'.format(score))

        # Save results
        # results = {
        #     # 'predictions': predictions,
        #     # 'pred_proba': pred_proba,
        #     'scores': score,
        #     # 'threshold': threshold,
        #     # 'idxs': (train_idx, test_idx)
        # }

        return score

    def __call__(self, train_idx, test_idx):
        results = self.prediction_loop(train_idx, test_idx)
        return results

