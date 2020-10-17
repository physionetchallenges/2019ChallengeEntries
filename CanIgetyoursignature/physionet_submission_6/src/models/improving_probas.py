""" Here we will use the output probas/regressors to generate more features and recompute """
from definitions import *
from lightgbm import LGBMRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.models.functions import CustomStratifiedGroupKFold, cross_val_predict_to_series, load_munged_data
from src.models.optimizers import ThresholdOptimizer
from src.models.evaluators import ComputeNormalizedUtility

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Raw data
_, labels_binary, labels_utility = load_munged_data()

# Get probas and print initial score
probas = load_pickle(MODELS_DIR + '/experiments/main/dd/1/probas.pickle')
_, score_init = ThresholdOptimizer().optimize(labels_binary, probas)
ppprint('Initial score: {:.3f}'.format(score_init))


# Now compute some extra features from the probabilities
def func(df):
    std = df.rolling(10, min_periods=10).std()
    mean = df.rolling(10, min_periods=10).mean()
    max = df.rolling(10, min_periods=10).max()
    return pd.concat([std, mean, max], axis=1)

new_features = groupby_apply_parallel(probas.groupby('id'), func)

# Now run a second model
data = pd.concat([probas, new_features], axis=1)
cv_iter = CustomStratifiedGroupKFold(n_splits=5).split(data, labels_binary, groups=data.index.get_level_values('id'))
new_pred = cross_val_predict_to_series(LGBMRegressor(), data, labels_utility, cv=cv_iter)

# Optimise thresh
t, score = ThresholdOptimizer().optimize(labels_binary, new_pred)
ppprint('New score w/o changing: {:.3f}'.format(score))

# Change only those with T > 10
idx = pd.IndexSlice[:, 10:]
probas.loc[idx] = new_pred.loc[idx]
t, final_score = ThresholdOptimizer().optimize(labels_binary, probas)
ppprint('Score for T > 10 score: {:.3f}'.format(final_score))

# Change only those with T > 10
idx = pd.IndexSlice[:, 20:]
probas.loc[idx] = new_pred.loc[idx]
t, final_score = ThresholdOptimizer().optimize(labels_binary, probas)
ppprint('Score for T > 20: {:.3f}'.format(final_score))

