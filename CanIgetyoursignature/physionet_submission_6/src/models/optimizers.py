from definitions import *
import multiprocessing
# from nevergrad.optimization import optimizerlib
# from nevergrad import instrumentation as inst
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from lightgbm import LGBMRegressor

# My imports
from src.models.evaluators import *
from src.models.functions import *
from src.data.transformers import *

# Annoying pandas warns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ThresholdOptimizer():
    """
    Given labels and proba or regression predictions, finds the optimal threshold to attain the max utility score.
    """
    def __init__(self, labels=None, preds=None, budget=200, parallel=False, cv_num=False, jupyter=False):
        self.labels = labels
        self.preds = preds
        self.budget = budget
        self.num_workers = 1 if parallel is False else multiprocessing.cpu_count()
        self.cv_num = cv_num

        # Get the scores
        self.scores_loc = DATA_DIR + '/processed/labels/full_scores.pickle' if jupyter is False else ROOT_DIR + '/data/processed/labels/full_scores.pickle'
        self.scores = load_pickle(self.scores_loc)

    @staticmethod
    def score_func(scores, predictions, inaction_score, perfect_score, thresh=0):
        """ The utility score function, scores and predictions must be entered as numpy arrays. """
        # Apply the threshold
        predictions = (predictions > thresh).astype(int)

        # Get the actual score
        actual_score = scores[:, 1][predictions == 1].sum() + scores[:, 0][predictions == 0].sum()

        # Get the normalized score
        normalized_score = (actual_score - inaction_score) / (perfect_score - inaction_score)

        return normalized_score

    def optimize(self, labels, predictions):
        """ Main function for optimization of a threshold given labels and predictions. """
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        # We only want scores correspondent with labels
        scores = self.scores.loc[labels.index].values

        # Give bounds
        instrum = inst.Instrumentation(*[inst.var.Array(1).asscalar().bounded(-0.2, 0.2)])

        # Set optimizer
        optimizer = optimizerlib.TwoPointsDE(instrumentation=instrum, budget=self.budget, num_workers=self.num_workers)

        # Precompute the inaction and perfect scores
        inaction_score = scores[:, 0].sum()
        perfect_score = scores[:, [0, 1]].max(axis=1).sum()

        # Optimize
        recommendation = optimizer.optimize(
            lambda thresh: -self.score_func(scores, predictions, inaction_score, perfect_score, thresh=thresh)
        )

        # Get the threshold and return the score
        threshold = recommendation.args[0]
        score = self.score_func(scores, predictions, inaction_score, perfect_score, thresh=threshold)

        return threshold, score

    def cv_func(self, train_idx, test_idx, cv_num=False):
        """ The function run for each fold of the cross_val_threshold method. """
        # Split test train
        labels_train, labels_test = self.labels.iloc[train_idx], self.labels.iloc[test_idx]
        preds_train, preds_test = self.preds.iloc[train_idx], self.preds.iloc[test_idx]

        # Optimise the training
        threshold, score = self.optimize(labels_train, preds_train)

        # Apply to the testing
        preds_test_thresh = (preds_test >= threshold).astype(int)
        test_score = ComputeNormalizedUtility().score(labels_test, preds_test_thresh, cv_num=cv_num)
        ppprint('\tScore on cv fold: {:.3f}'.format(test_score))

        return preds_test_thresh, test_score, threshold

    @timeit
    def cross_val_threshold(self, cv, give_cv_num=False, parallel=True):
        """
        Similar to cross val predict, performs the thresholding algorithm on the given cv folds.

        Note that if this is specified, labels and pred must be preloaded in __init__()
        """
        results = parallel_cv_loop(self.cv_func, cv, give_cv_num=give_cv_num, parallel=parallel)

        # Open out results
        preds = pd.concat([x[0] for x in results], axis=0)
        scores = [x[1] for x in results]
        thresholds = [x[2] for x in results]

        return preds, scores, thresholds


class LGBMTuner():
    """
    Class for tuning the parameters of an LGBM class.
    """
    def __init__(self, scoring='accuracy'):
        self.score_history = []
        self.scoring = scoring
        self.clf = XGBRegressor(n_estimators=1)

        # Normal GS
        self.depth_and_leaves = {
            'max_depth': [2, 4, 6, 8, 10],
            'num_leaves': [20, 30, 40, 50, 60, 70],
        }
        self.child_params = {
            'min_child_weight': [1, 3, 5, 7, 10, 15, 20],
            'min_child_samples': [10, 50, 100, 150, 200, 300, 350, 400],
        }
        self.subsample_colsample = {
            'subsample': [0.01, 0.1, 0.3, 0.5, 0.8],
            'colsample_bytree': [0.01, 0.1, 0.3, 0.5, 0.8],
        }
        self.reg = {
            'reg_alpha': [0, 0.1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda': [0, 0.1, 1, 5, 10, 20, 50, 100]
        }

    def score(self, labels, predictions):
        """ Score after a threshold optimization. """
        # Get the score and update the history
        _, score = ThresholdOptimizer().optimize(labels, predictions)
        self.score_history.append(score)

        # Print information
        print('Score: {}'.format(score))
        if len(self.score_history) > 1:
            print('Gain: {}'.format(score - self.score_history[-2]))

    def tune_params(self, params, data, labels, cv):
        """ Basic function for gridsearch tuning parameters """
        gs = GridSearchCV(
            estimator=self.clf, param_grid=params, cv=cv, scoring=make_scorer(gs_utility_eval), n_jobs=-1
        )
        gs.fit(data, labels)

        # Update the params of the classifier
        self.clf.set_params(**gs.best_params_)

        # CV predict using the best estimator and return the score
        preds = cross_val_predict(gs.best_estimator_, data, labels, cv=cv)
        self.score(labels, preds)

        return gs.best_params_

    def fine_tune_params(self, params, data, labels, cv=None):
        pass

    def tune(self, data, labels, cv=5):
        # Begin
        print('START:')
        predictions = cross_val_predict(self.clf, data, labels, cv=cv)
        self.score(labels, predictions)

        # Tune depth and leaves
        print('\nTuning max_depth and num_leaves')
        params = self.tune_params(self.depth_and_leaves, data, labels, cv=cv)
        print('~Fine tuning~')
        self.fine_tune_params(params, data, labels, cv=cv)

        # Tune child
        print('\nTuning max_depth and num_leaves')
        self.tune_params(self.child_params, data, labels)
        print('~Fine tuning~')

        # Tune subsample colsample
        print('\nTuning subsample and colsample_bytree')
        self.tune_params(self.subsample_colsample, data, labels)
        print('~Fine tuning~')

        # Tune
        print('\nTuning regularization')
        self.tune_params(self.reg, data, labels)
        print('~Fine tuning~')

        # Print, save and return the best parameters
        print('Best params: {}'.format(self.clf.get_params()))
        self.best_params = self.clf.get_params()
        return self.best_params


def random_lgbm_tuner(clf, data, labels, cv=None, param_grid=None, n_iter=10):
    """ Performs RandomCV tuning for the LGBM class. """
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'num_leaves': sp_randint(20, 80),
            'min_child_samples': sp_randint(20, 200),
            'min_child_weight': [0.1, 1, 10, 20, 35, 50, 75, 100],
            'subsample': sp_uniform(loc=0.2, scale=0.8),
            'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
            'reg_alpha': [0, 0.1, 1, 5, 10, 50, 100],
            'reg_lambda': [0, 0.1, 1, 5, 10, 20, 50, 100],
        }

    # Run the gridsearch
    gs = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        cv=cv,
        n_iter=n_iter,
        scoring=make_scorer(gs_utility_eval, greater_is_better=True),
        n_jobs=8,
        refit=False,
        verbose=True,
    )
    gs.fit(data, labels)

    # Print info and return the params
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

    return gs.best_params_


def lgbm_utility_eval(preds, train_data):
    """ Write up of the utility_function as an lgbm eval function. """
    labels = train_data.get_label()
    t, s = ThresholdOptimizer().optimize(labels, preds)
    return 'utility_score', s, True


def gs_utility_eval(labels, preds):
    """ Write up of the utility_function as an sklearn gs function. """
    t, s = ThresholdOptimizer().optimize(labels, preds)
    return s


# PARAMS USED IN GOOD SUBMISSION
old_params = {
    'boosting_type': 'gbdt',
    'class_weight': None,
    'colsample_bytree': 0.5493559885258208,
    'importance_type': 'split',
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_samples': 122,
    'min_child_weight': 1,
    'min_split_gain': 0.0,
    'n_estimators': 1000,
    'n_jobs': -1,
    'num_leaves': 49,
    'objective': None,
    'random_state': None,
    'reg_alpha': 100,
    'reg_lambda': 0,
    'silent': True,
    'subsample': 0.3465044484531902,
    'subsample_for_bin': 200000,
    'subsample_freq': 0
}


if __name__ == '__main__':
    # Get data and cv
    _, labels_binary, labels_utility = load_munged_data()
    df = load_pickle(DATA_DIR + '/processed/dataframes/df_full.pickle')
    cv = CustomStratifiedGroupKFold(seed='random').split(df, labels_binary)
    clf = LGBMRegressor(n_estimators=100)

    # Random tune
    best_params = random_lgbm_tuner(clf, df, labels_utility, cv, n_iter=3000)
    save_pickle(best_params, MODELS_DIR + '/parameters/lgb/random_grid_fullds.pickle')
