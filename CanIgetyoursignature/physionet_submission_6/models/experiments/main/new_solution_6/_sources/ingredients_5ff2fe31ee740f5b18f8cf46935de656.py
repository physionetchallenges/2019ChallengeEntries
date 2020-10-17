from sacred import Ingredient
data_ingredient = Ingredient('data')
feature_ingredient = Ingredient('feature')
train_ingredient = Ingredient('train')

# My imports
from lightgbm import LGBMRegressor
from src.models.experiments.functions import *
from src.features.transformers import *
from src.models.functions import *
from src.models.experiments.train_model import *
from src.features.feature_selection import FeatureSelector
from src.data.extracts import irregular_cols, cts_cols
from src.models.optimizers import old_params


# Feature Generation
@feature_ingredient.config
def feature_config():
    # Use RFECV feature selection
    feature_selction = False

    # Num measurements in the last n_hours
    num_measurements = False

    # Moments
    moments = False
    moment_lookback = 6

    # Signature options
    columns = False
    lookback = False
    lookback_method = 'mean'
    individual = False
    order, logsig, leadlag, addtime, cumsum, pen_off, append_zero = 2, True, False, False, False, False, False

    # Cumsum signatures
    cumsum_columns = False
    cumsum_addtime = False
    cumsum_lookback = 10
    cumsum_order = 3

    # Other
    extra_moments = False
    add_max, add_min = False, False
    max_min_lookback = 5
    drop_count = False

    # For submission
    last_only = False

@feature_ingredient.capture
def generate_features(_run, feature_selection,
    num_measurements, moments, moment_lookback,
    columns, lookback, lookback_method, individual, order, logsig, leadlag, addtime, cumsum, pen_off, append_zero,
    extra_moments, add_max, add_min, max_min_lookback, drop_count,
    last_only):
    # Get data
    df, labels_binary, labels_utility = load_munged_data()
    df.drop('hospital', axis=1, inplace=True)

    # Get number of measurements taken in a fixed time window
    counts_24hrs = None
    if num_measurements is not False:
        cols = [x for x in df.columns if '_count' in x]
        counts_24hrs = GetNumMeasurements(lookback=num_measurements).transform(df[cols])
        counts_24hrs = numpy_to_named_dataframe(counts_24hrs, df.index, 'cntxhrs')

    # Moments
    moments_frame = None
    if moments is not False:
        moments_frame = AddMoments(moments=moments, lookback=moment_lookback, last_only=last_only).transform(df)
        moments_frame = numpy_to_named_dataframe(moments_frame, df.index, 'Moments')
        moments_frame.columns = ['{}_moment_{}'.format(col, i) for i in range(2, moments + 1) for col in df.columns]

    # Add signatures
    signatures = None
    if columns is not False:
        signatures = add_signatures(df, columns, individual, lookback, lookback_method,
                                    order, logsig, leadlag, addtime, cumsum, pen_off, append_zero, last_only=last_only)
        signatures = numpy_to_named_dataframe(signatures, df.index, 'Signatures')

    # Get sampling rate rather than absolute number for the count column.
    data = GetRateOfLaboratorySampling().transform(df)

    # Get extra moments
    new_moments = None
    if extra_moments is not False:
        cols = ['HCO3', 'HR']
        new_moments = AddMoments(moments=extra_moments, start=4, force_compute=True).transform(df[cols])
        new_moments = pd.DataFrame(index=df.index, data=new_moments)
        new_moments.columns = ['{}_moment_{}'.format(col, i) for i in range(4, extra_moments + 1) for col in cols]

    # Max and min
    max_vals = None
    if add_max is not False:
        max_vals = GetStatistic(statistic='max', lookback=max_min_lookback, columns=cts_cols).transform(df[cts_cols])
        max_vals = pd.DataFrame(index=df.index, data=max_vals, columns=['{}_max'.format(x) for x in cts_cols])

    min_vals = None
    if add_min is not False:
        min_vals = GetStatistic(statistic='min', lookback=max_min_lookback, columns=cts_cols).transform(df[cts_cols])
        min_vals = pd.DataFrame(index=df.index, data=min_vals, columns=['{}_min'.format(x) for x in cts_cols])

    # Create data ready for insertion
    # df.drop([x for x in df.columns if '_count' in x], axis=1, inplace=True)
    df = pd.concat([data, counts_24hrs, moments_frame, signatures, new_moments, max_vals, min_vals], axis=1)
    # data = np.concatenate([x for x in (data, counts_24hrs, moments_frame, signatures) if x is not None], axis=1)
    # df = pd.DataFrame(index=df.index, data=data)
    # df = remove_useless_columns(df)

    # Try RFECV
    if feature_selection is not False:
        if feature_selection == 'from_save':
            features = load_pickle(MODELS_DIR + '/feature_selection/rfecv_test.pickle')
            df = df[[x for x in features if x in df.columns]]
        else:
            fs = FeatureSelector(verbose=1).fit(df, labels_utility)
            df = fs.transform(df)

    # Drop the count col
    if drop_count:
        df.drop([x for x in df.columns if '_count' in x], axis=1, inplace=True)

    # Print info about the shape
    print(df.shape)
    print(df.columns)

    # Add to run
    _run.df, _run.labels_binary, _run.labels_utility = df, labels_binary, labels_utility

    # Save for checking solution runs
    save_pickle(df, DATA_DIR + '/processed/dataframes/df_full.pickle')

    # For training early time
    df = df[df.columns[0:40]]
    print('THIS IS THE ACTUAL SHAPE', df.shape)

    return df

# TRAIN
@train_ingredient.config
def train_config():
    gs_params = False
    n_estimators = 100
    learning_rate = 0.1

@train_ingredient.capture
def train_model(_run, gs_params, n_estimators, learning_rate):
    # Load
    df, labels_binary, labels_utility = _run.df, _run.labels_binary, _run.labels_utility

    # Get the cross validated folds
    cv_iter = CustomStratifiedGroupKFold(n_splits=5).split(df, labels_binary, groups=df.index.get_level_values('id'))

    # Setup the training loop function
    if gs_params is not False:
        params = load_pickle(MODELS_DIR + '/parameters/lgb/random_grid_fullds.pickle')
        # params = load_pickle(MODELS_DIR + '/parameters/lgb/random_grid.pickle')
        print(params)
        clf = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate).set_params(**params)
    else:
        clf = LGBMRegressor(**old_params).set_params(**{'n_estimators': n_estimators, 'learning_rate': learning_rate})
    predictions = cross_val_predict_to_series(clf, df, labels_utility, cv=cv_iter, n_jobs=-1)

    # Perform thresholding
    binary_preds, scores, _ = ThresholdOptimizer(budget=100, labels=labels_binary, preds=predictions).cross_val_threshold(cv_iter, parallel=True, give_cv_num=True)

    # Log info
    ppprint('AVERAGE SCORE {:.3f}'.format(np.mean(scores)), color='green')
    _run.log_scalar('utility_score', np.mean(scores))
    save_pickle(predictions, _run.save_dir + '/probas.pickle')

