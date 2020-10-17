"""
Here we train an LSTM model on the previous probabilities to get a better prediction at the current timepoint.
"""
from definitions import *
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing import sequence
from src.models.functions import CustomStratifiedGroupKFold
from src.features.transformers import MakePaths
from src.models.optimizers import ThresholdOptimizer
from src.models.evaluators import ComputeNormalizedUtility


# Get labels and probas
labels = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')
probas = load_pickle(MODELS_DIR + '/experiments/main/new_solution/1/probas.pickle')

# Make the data into the correct form
n_steps = 10
n_features = 1
paths = np.array(MakePaths(lookback=n_steps - 1, method='fixed').transform(probas))

# Setup the model
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Set CV
cv = CustomStratifiedGroupKFold().split(probas, labels)
train_idx, test_idx = cv[0]

#
# # Split train test
# X_train, X_test = paths[train_idx], paths[test_idx]
# y_train, y_test = labels.iloc[train_idx].values, labels.iloc[test_idx].values
#
#
# # Print original score
# t, s = ThresholdOptimizer().optimize(labels.iloc[train_idx], probas.iloc[train_idx])
# ppprint('Original score on training data: {:.3f}'.format(s))
#
# t, s = ThresholdOptimizer().optimize(labels.iloc[test_idx], probas.iloc[test_idx])
# ppprint('Original score on test data: {:.3f}'.format(s))
#
# # Initialize variables
# timesteps = paths.shape[1]
# n_features = 1
#
# epochs = 200
# batch = 64
# lr = 0.01
#
# history = model.fit(X_train, y_train, epochs=2, batch_size=2, verbose=2)
#
# # Predict
# train_preds = model.predict(X_train).reshape(-1)
# test_preds = model.predict(X_test).reshape(-1)
#
# # New score on training data
# t, s = ThresholdOptimizer().optimize(labels.iloc[train_idx], train_preds)
# ppprint('New score on training data: {:.3f}'.format(s))
#
# # New score on test data
# t, s = ThresholdOptimizer().optimize(labels.iloc[test_idx], test_preds)
# ppprint('New score on test data: {:.3f}'.format(s))

