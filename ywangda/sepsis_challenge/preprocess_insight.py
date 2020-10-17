import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from util import *
import time
from pandas import concat
from shutil import copyfile


def split_normal_and_sepsis(data_dir, normal_dir, sepsis_dir):
    print('split the data into normal group and sepsis group...')
    files = glob.glob(data_dir + '*.psv')
    for file_name in files:
        if file_name.endswith('.psv'):
            (values, column_names) = read_raw_data(file_name)
            df = pd.DataFrame(values, columns=column_names)
            # normal patient
            if df['SepsisLabel'].max() == 0:
                # copy file
                copyfile(file_name, normal_dir+file_name[len(data_dir):])
            else:
                copyfile(file_name, sepsis_dir + file_name[len(data_dir):])


def compute_diff_median_and_mean(data_dir, dynamic_column_names, n_in):
    print('compute difference median...')
    # data_dir = 'D:/Projects/physionet-master/training_data/training_stage1/'
    # normal_dir = 'D:/Projects/physionet-master/training_data/normal_group/'
    # sepsis_dir = 'D:/Projects/physionet-master/training_data/sepsis_group/'
    files = glob.glob(data_dir + '*.psv')
    n_patients = len(files)                                        # 45336
    n_file = 0
    n_normal_record = 0
    n_sepsis_record = 0
    print('patient no:', n_patients)

    # dynamic_column_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'pH', 'WBC']
    cols_diff, cols_basic, names = list(), list(), list()
    for file_name in files:
        if file_name.endswith('.psv'):
            (values, column_names) = read_raw_data(file_name)
            df = pd.DataFrame(values, columns=column_names)
            # normal patient
            if df['SepsisLabel'].max() == 0:
                df = df[dynamic_column_names]

                # impute missing value by forward and backward filling in
                # df.fillna(method='ffill', inplace=True)
                # df.fillna(method='bfill', inplace=True)

                cols_basic.append(df)
                cols_diff.append(df.diff(periods=n_in))
                # copy file
                # copyfile(file_name, normal_dir + file_name[len(data_dir):])
                n_normal_record += df.shape[0]
            # sepsis patient
            else:
                # copyfile(file_name, sepsis_dir + file_name[len(data_dir):])
                n_normal_record += df.shape[0] - df['SepsisLabel'].sum()
                n_sepsis_record += df['SepsisLabel'].sum()

            n_file += 1
            if n_file % 1000 == 0:
                print('read file: ', n_file)

    print('normal record: ', n_normal_record, 'sepsis record: ', n_sepsis_record)
    basic = pd.concat(cols_basic, axis=0)
    basic.columns = dynamic_column_names
    f = open("median_basic", "wb")
    pickle.dump(pd.DataFrame(basic.median(axis=0, skipna=True)), f)
    f.close()
    f = open("mean_basic", "wb")
    pickle.dump(pd.DataFrame(basic.mean(axis=0, skipna=True)), f)
    f.close()

    reference = concat(cols_diff, axis=0)
    reference.columns = dynamic_column_names
    f = open("median_diff", "wb")
    pickle.dump(pd.DataFrame(reference.median(axis=0, skipna=True)), f)
    f.close()
    f = open("mean_diff", "wb")
    pickle.dump(pd.DataFrame(reference.mean(axis=0, skipna=True)), f)
    f.close()
    return


def bin_difference(df, dynamic_col_name, mean):
    mean = mean.to_dict()[0]
    df_bin = pd.DataFrame()
    for i in dynamic_col_name:
        # a = pd.cut(df[i], [-np.inf, -np.abs(mean[i]), np.abs(mean[i]), np.inf], [-1, 0, 1])
        df_bin[i + '_trend'] = pd.cut(df[i], bins=[-np.inf, -np.abs(mean[i]), np.abs(mean[i]), np.inf], labels=[-1, 0, 1])
    return df_bin.astype(np.float64)


def bin_corr(df, corr_name):
    df_bin = pd.DataFrame()
    for i in corr_name:
        df_bin[i] = pd.cut(df[i], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[-1, 0, 1])
    return df_bin.astype(np.float64)


def bin_tri_corr(df, triplet_corr_name):
    df_bin = pd.DataFrame()
    for i in triplet_corr_name:
        df_bin[i] = pd.cut(df[i], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[-1, 0, 1])
    return df_bin.astype(np.float64)


def triplet_corr(df, time_win):
    '''
    df is a three-column dataframe with each column the correlation coefficient.
    '''

    def f(x):
        x0 = x[0]-np.mean(x[0])
        x1 = x[1]-np.mean(x[1])
        x2 = x[2]-np.mean(x[2])
        x = np.sum(x0*x1*x2)
        v0 = np.std(x[0])
        v1 = np.std(x[1])
        v2 = np.std(x[2])
        return x/(v0*v1*v2+1e-6)      # prevent divided by zero

    triplet_corr = df.rolling(time_win, on = [df.columns]).apply(f)
    return triplet_corr


def generate_minimal_feature(df, dynamic_col_name,  static_col_name, time_win):
    '''
    use pd.shift function
    '''
    print('generating minimal feature...')
    cols, names = list(), list()
    # input sequence (t-time_win, ... t-1)
    for i in range(time_win, 0, -1):
        cols.append(df[dynamic_col_name].shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in dynamic_col_name]

    cols.append(df[dynamic_col_name])
    names += [('%s(t)' % j) for j in dynamic_col_name]

    diff_names = list()
    diff_df = df[dynamic_col_name].diff(periods=time_win)
    diff_df.fillna(method='ffill', inplace=True)
    diff_df.fillna(method='bfill', inplace=True)

    diff_names += [('%s_diff' % j) for j in dynamic_col_name]
    cols.append(diff_df)
    names += [('%s_diff' % j) for j in dynamic_col_name]

    cols.append(df[static_col_name])
    names += [('%s' % j) for j in static_col_name]

    # concatenate the columns
    minimal = concat(cols, axis=1)
    minimal.columns = names
    # fill nan
    minimal.fillna(method='ffill', inplace=True)
    minimal.fillna(method='bfill', inplace=True)

    print('minimal feature shape:', minimal.shape)
    return minimal


def generate_basic_feature(df, dynamic_col_name,  static_col_name, time_win):
    '''
    use pd.shift function
    '''
    print('generating basic feature...')
    cols, names = list(), list()
    # input sequence (t-time_win, ... t-1)
    for i in range(time_win, 0, -1):
        cols.append(df[dynamic_col_name].shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in dynamic_col_name]

    cols.append(df[dynamic_col_name])
    names += [('%s(t)' % j) for j in dynamic_col_name]

    cols.append(df[dynamic_col_name].rolling(time_win).mean())
    names += [('%s_mean' % j) for j in dynamic_col_name]

    cols.append(df[dynamic_col_name].rolling(time_win).min())
    names += [('%s_min' % j) for j in dynamic_col_name]

    cols.append(df[dynamic_col_name].rolling(time_win).max())
    names += [('%s_max' % j) for j in dynamic_col_name]

    cols.append(df[static_col_name])
    names += [('%s' % j) for j in static_col_name]

    # concatenate the columns
    basic = concat(cols, axis=1)
    basic.columns = names
    # fill nan
    basic.fillna(method='ffill', inplace=True)
    basic.fillna(method='bfill', inplace=True)

    print('basic feature shape:', basic.shape)
    return basic


def generate_reference_feature(df, dynamic_col_name, static_col_name, mean_diff, time_win):
    '''
    use pd.rolling function
    '''
    print('generating reference feature...')
    # dynamic_column_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'pH', 'WBC']
    # static_column_names = ['Age', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
    n_dynamic_feature = len(dynamic_col_name)

    cols, names = list(), list()
    diff_names = list()
    diff_df = df[dynamic_col_name].diff(periods=time_win)
    diff_df.fillna(method='ffill', inplace=True)
    diff_df.fillna(method='bfill', inplace=True)

    diff_names += [('%s_diff' % j) for j in dynamic_col_name]
    cols.append(diff_df)
    names += [('%s_diff' % j) for j in dynamic_col_name]

    trend_names = list()
    trend_df = bin_difference(diff_df, dynamic_col_name, mean_diff)
    trend_df.fillna(method='ffill', inplace=True)
    trend_df.fillna(method='bfill', inplace=True)
    trend_names += [('%s_trend' % j) for j in dynamic_col_name]
    cols.append(trend_df)
    names += [('%s_trend' % j) for j in dynamic_col_name]

    corr_list = list()
    corr_names = list()
    for i in range(n_dynamic_feature):
        for j in range(i+1, n_dynamic_feature):
            # corr_col = diff_df[i].rolling(time_win).corr(diff_df[j])
            corr_list.append(diff_df[dynamic_col_name[i]].rolling(time_win).corr(diff_df[dynamic_col_name[j]]))
            corr_names += [('%s_%s_corr' % (dynamic_col_name[i], dynamic_col_name[j]))]
            names += [('%s_%s_corr' % (dynamic_col_name[i], dynamic_col_name[j]))]

    corr_df = pd.concat(corr_list, axis=1)
    corr_df.columns = corr_names
    corr_df.fillna(method='ffill', inplace=True)
    corr_df.fillna(method='bfill', inplace=True)
    corr_df.fillna(0, inplace=True)
    bin_corr_df = bin_corr(corr_df, corr_names)
    bin_corr_df.fillna(method='ffill', inplace=True)
    bin_corr_df.fillna(method='bfill', inplace=True)
    bin_corr_df.fillna(0, inplace=True)
    cols.append(bin_corr_df)
    '''
    # multiple correlation coefficient
    tri_corr_list, tri_corr_names = list(), list()
    for i in range(n_dynamic_feature):
        for j in range(i+1, n_dynamic_feature):
            for k in range(j+1, n_dynamic_feature):
                tri_corr_list.append(triplet_corr(diff_df[[dynamic_col_name[i], dynamic_col_name[j], dynamic_col_name[k]]], time_win))
                tri_corr_names += [('%s_%s_%s_corr' % (dynamic_col_name[i], dynamic_col_name[j], dynamic_col_name[k]))]
                names += [('%s_%s_%s_corr' % (dynamic_col_name[i], dynamic_col_name[j], dynamic_col_name[k]))]
    tri_corr_df = pd.concat(tri_corr_list, axis=1)
    tri_corr_df.columns = tri_corr_names
    tri_corr_df.fillna(method='ffill', inplace=True)
    tri_corr_df.fillna(method='bfill', inplace=True)
    corr_df.fillna(0, inplace=True)
    bin_tri_corr_df = bin_tri_corr(tri_corr_df, tri_corr_names)
    cols.append(bin_tri_corr_df)
    '''

    cols.append(df[static_col_name])
    names += [('%s' % j) for j in static_col_name]

    reference = concat(cols, axis=1)         # concatenate columns
    reference.columns = names
    print('reference feature shape:', reference.shape)
    return reference


def preprocess_data(data_dir, feature, vital_column_names, dynamic_column_names, static_column_names, time_win, scaler_name, mean_basic, mean_diff):
    # split the dataset
    files = glob.glob(data_dir + '*.psv')
    patients_no = len(files)                                       # 45336
    # patients_train = 35336
    # patients_test = 10000
    # record_no = 1740663
    # record_train = 1357085
    # record_test = record_no - record_train
    # sepsis = np.zeros(patients_no) - 1
    file_no = 0
    feat_list = list()

    for file_name in files:
        if file_name.endswith('.psv'):
            file_no += 1
            print('-----------------------------------------------------------------------')
            print('preprocessing file: ', file_name, ' process: ', file_no, '/', patients_no)
            (values, column_names) = read_raw_data(file_name)
            df = pd.DataFrame(values, columns=column_names)
            print('input feature shape:', df.shape)
            vital_feat_df = df[vital_column_names]                     # select vital feature

            # impute missing value by forward, backward and median filling in
            vital_feat_df.fillna(method='ffill', inplace=True)
            vital_feat_df.fillna(method='bfill', inplace=True)
            vital_feat_df.fillna(mean_basic.to_dict()[0], inplace=True)      # convert dataframe to dict

            # construct the feature
            if feature == 'minimal':
                df = generate_minimal_feature(vital_feat_df, dynamic_column_names, static_column_names, time_win)
            if feature == 'basic':
                df = generate_basic_feature(vital_feat_df, dynamic_column_names, static_column_names, time_win)
            if feature == 'reference':
                df = generate_reference_feature(vital_feat_df, dynamic_column_names, static_column_names, mean_diff, time_win)

            # if basic_df.shape[0] != reference_df.shape[0]:
            #     print(basic_df.shape[0], basic_df.shape[1])
            feat_list.append(df)

    # combine the data
    print('combining files...')
    t = time.time()
    feat_df = pd.concat(feat_list, axis=0)
    print('elapsed time:', time.time()-t)
    print('feat shape:', feat_df.shape)

    # ensure all data is float
    values = feat_df.values  # returns a numpy array
    values = values.astype('float32')

    # normalize features
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler1.fit_transform(values)
    feat_df = pd.DataFrame(X_scaled, columns=feat_df.columns)
    joblib.dump(scaler1, 'minmax_scaler.save')

    # normalize features
    scaler2 = RobustScaler()
    X_scaled = scaler2.fit_transform(values)
    feat_df = pd.DataFrame(X_scaled, columns=feat_df.columns)
    joblib.dump(scaler2, 'robust_scaler.save')

    # handle nan
    feat_df.dropna(inplace=True)
    print('dropna feat shape:', feat_df.shape)

    # save to csv
    feat_df.to_csv('D:/Projects/physionet-master/training_data/' + feature + '_df.csv', index=False)


if __name__ == '__main__':
    train_dir = 'D:/Projects/physionet-master/training_data/train/'
    normal_dir = 'D:/Projects/physionet-master/training_data/normal_group/'
    sepsis_dir = 'D:/Projects/physionet-master/training_data/sepsis_group/'
    # split_normal_and_sepsis(data_dir, normal_dir, sepsis_dir)

    vital_column_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC', 'Age', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
    dynamic_column_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC']
    static_column_names = ['Age', 'HospAdmTime', 'ICULOS', 'SepsisLabel']

    scaler_name = "scaler.save"
    feature = 'minimal'      # 'basic' , 'reference'
    time_win = 4         # t-4, t-3, t-2, t-1, t
    # compute_diff_median_and_mean(data_dir, dynamic_column_names, 5)

    f = open('mean_basic', 'rb')
    mean_basic = pickle.load(f)
    print(mean_basic)

    f = open('mean_diff', 'rb')
    mean_diff = pickle.load(f)
    print(mean_diff)

    preprocess_data(normal_dir, feature, vital_column_names, dynamic_column_names, static_column_names, time_win, scaler_name, mean_basic, mean_diff)