import sys
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from util import *
import time


def read_column_name(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_name = header.split('|')
    return column_name


def read_raw_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_name = header.split('|')
        value = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    '''
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    '''
    return value, column_name


def impute_by_mean(df, column_names):
    # impute data by mean
    meaner = df.mean()
    df.fillna(df.mean(), inplace=True)
    return df, meaner


def scale_data(df, column_names):
    idx = np.argwhere(column_names == 'SepsisLabel')
    column_names = np.delete(column_names, idx)
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    df[column_names] = scaler.fit_transform(df[column_names])       # exclude the label
    return df, scaler


def compute_column_mean(df, column_names):
    # Compute the arithmetic mean along the specified axis, ignoring NaNs.
    column_mean_df = df[column_names].mean()
    '''
    train_num = 150596  # records of first 4000 patients
    test_num = 188453 - 150596  # records of last 1000 patients
    train_data = raw_df[column_names].values[:train_num]
    column_mean = np.nanmean(train_data, axis=0)   # only use training data to compute mean
    column_mean = column_mean.reshape((1, len(column_names)))
    '''
    print(column_mean_df)
    return column_mean_df


def impute_missing_value(column_mean_df, column_names, files):
    columns_num = len(column_names)
    column_mean_dict = column_mean_df.to_dict()
    print(column_mean_dict)
    # combine psv files
    combined_values = np.array([], dtype=np.float64).reshape(0, columns_num)  # include the label
    combined_imputed_df = pd.DataFrame(combined_values, columns=column_names)

    for file_name in files:
        if file_name.endswith('.psv'):
            (values, _) = read_challenge_data(file_name)   # colum_name is not used
            df = pd.DataFrame(values, columns=column_names)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            combined_imputed_df = pd.concat([combined_imputed_df, df])

    nan_num = combined_imputed_df.isnull().sum()
    record_num = combined_imputed_df.shape[0]
    print('record_num: ', record_num)
    dropped_columns = column_names[nan_num > 0.5 * record_num]
    print(dropped_columns)

    combined_imputed_df = combined_imputed_df.fillna(value=column_mean_dict)
    combined_imputed_df.to_csv('D:/Projects/physionet-master/training_data/full_imputed_data.csv', index=False)
    dropped_imputed_data = combined_imputed_df.drop(dropped_columns, axis=1)
    dropped_imputed_data.to_csv('D:/Projects/physionet-master/training_data/dropped_imputed_data.csv', index=False)
    return nan_num


def construct_feature(df):
    #  must call after filling in
    ####################################################################################################################
    # SIRS (old standard)
    ####################################################################################################################
    # (1) Temperature >38°C or <36°C
    def filter_sirs_temprature(temp):
        if temp > 38.0 or temp < 36.0:
            return 1
        else:
            return 0
    df['sirs_temp'] = df['Temp'].apply(filter_sirs_temprature)

    # (2) Heart rate > 90/min
    def filter_sirs_hr(hr):
        if hr > 90.0:
            return 1
        else:
            return 0
    df['sirs_hr'] = df['HR'].apply(filter_sirs_hr)

    # (3) Respiratory rate    20/min
    def filter_sirs_resp(resp):
        if resp > 20.0:
            return 1
        else:
            return 0
    df['sirs_resp'] = df['Resp'].apply(filter_sirs_resp)

    # (4) PaCO2 <32 mm Hg (4.3 kPa)
    def filter_sirs_paco2(paco2):
        if paco2 < 32.0:
            return 1
        else:
            return 0
    df['sirs_paco2'] = df['PaCO2'].apply(filter_sirs_paco2)

    # (5) White blood cell count >12 000/mm3 or <4000/mm3
    def filter_sirs_wbc(wbc):
        if wbc > 12.0 or wbc < 4.0:
            return 1
        else:
            return 0
    df['sirs_wbc'] = df['WBC'].apply(filter_sirs_wbc)

    # (6) SIRS score
    #def compute_sirs_score(df):
        #df.sum(axis=1, skipna=True)
        #return df
    df['sirs_score'] = df[['sirs_temp', 'sirs_hr', 'sirs_resp', 'sirs_paco2', 'sirs_wbc']].sum(axis=1, skipna=True)

    ####################################################################################################################
    # SOFA (new standard)
    ####################################################################################################################
    # (1) Respiratory, PO2/FiO2, mmHg (kPa)
    # no available PO2
    # (2) Coagulation, Platelets, ×103/mm3
    def filter_sofa_platelets(plate):
        if plate > 150.0:
            return 0
        elif plate > 100.0:
            return 1
        elif plate > 50.0:
            return 2
        elif plate > 20.0:
            return 3
        else:
            return 4
    df['sofa_platelets'] = df['Platelets'].apply(filter_sofa_platelets)

    # (3) Liver, Bilirubin, mg/dL
    def filter_sofa_bilirubin(bilir):
        if bilir < 1.2:
            return 0
        elif bilir < 1.9:
            return 1
        elif bilir < 5.9:
            return 2
        elif bilir < 11.9:
            return 3
        else:
            return 4
    df['sofa_bilirubin'] = df['Bilirubin_direct'].apply(filter_sofa_bilirubin)

    # (4) Cardiovascular, map
    def filter_sofa_map(map):
        if map > 70.0:
            return 0
        else:
            return 1
    df['sofa_map'] = df['MAP'].apply(filter_sofa_map)

    # (5) Central nervous system, Glasgow Coma Scale(GCS)
    # no available GCS

    # (6) Renal, Creatinine, mg/dL.
    def filter_sofa_creatinine(creati):
        if creati < 1.2:
            return 0
        elif creati < 1.9:
            return 1
        elif creati < 3.4:
            return 2
        elif creati < 4.9:
            return 3
        else:
            return 4
    df['sofa_creatinine'] = df['Creatinine'].apply(filter_sofa_creatinine)

    # (7) SOFA score
    df['sofa_score'] = df[['sofa_platelets', 'sofa_bilirubin', 'sofa_map', 'sofa_creatinine']].sum(axis=1, skipna=True)

    ####################################################################################################################
    # qSOFA
    ####################################################################################################################
    # (1) Respiratory rate ≥22/min
    def filter_qsofa_resp(resp):
        if resp > 22.0:
            return 1
        else:
            return 0
    df['qsofa_resp'] = df['Resp'].apply(filter_qsofa_resp)

    # (2) Change in mental status

    # (3) Systolic blood pressure ≤100 mmHg
    def filter_qsofa_sbp(sbp):
        if sbp < 100.0:
            return 1
        else:
            return 0
    df['qsofa_sbp'] = df['SBP'].apply(filter_qsofa_sbp)

    # (4) qSOFA score
    df['qsofa_score'] = df[['qsofa_resp', 'qsofa_sbp']].sum(axis=1, skipna=True)

    return df


####################################################################################################################
# compute changes in PTT & WBC  & Platelets & Temp & BUN & Alkalinephos & Creatinine
####################################################################################################################
# must call before filling in
def stat_measure(df, columns):
    ####################################################################################################################
    # ICULOS & HospAdmTime
    ####################################################################################################################
    # compute the end time of stay in ICU
    df['iculos_max'] = df['ICULOS'].max(axis=0, skipna=True)

    # compute the end time of stay in ICU
    df['iculos_admin_ratio'] = df['iculos_max']/(-df['HospAdmTime'] + np.eps)     # prevent divided by zero

    # compute the total time of stay in ICU
    df['iculos_admin_total'] = df['iculos_max']+(-df['HospAdmTime'])
    df['total_measure'] = 0
    df['accumulated_measure'] = 0
    rows = df.shape[0]

    for col in columns:
        for row in range(rows):
            # df[col + '_diff'] = df[0:row, col].diff(axis=0, periods=1)
            df.loc[row, col + '_accumulated_no'] = df.loc[0:row, col].count()                                           # count non-nan cells for each column
            df.loc[row, col + '_accumulated_exsit'] = np.where(df.loc[row, col + '_accumulated_no'] < 1, 0, 1)
            df.loc[row, col + '_accumulated_duplicate'] = np.where(df.loc[row, col + '_accumulated_no'] < 2, 0, 1)
            df.loc[row, col + '_accumulated_freq'] = df.loc[row, col + '_accumulated_no'] / df.loc[row, 'ICULOS']

        # df[col + '_diff'] = df[col].diff(axis=0, periods=1)
        df[col + '_no'] = df.loc[rows-1, col + '_accumulated_no']                                                       # count non-nan cells for each column
        df[col + '_exsit'] = df.loc[rows-1, col + '_accumulated_exsit']
        df[col + '_duplicate'] = df.loc[rows-1, col + '_accumulated_duplicate']
        df[col + '_freq'] = df.loc[rows-1, col + '_accumulated_freq']   # average measure freq
        df['accumulated_measure'] += df[col + '_accumulated_no']
    df['total_measure'] = df['accumulated_measure'].max()
    return df


def preprocess_pred(df, column_mean_dict):
    # stat the measure of important sign
    columns = ['WBC', 'Platelets', 'Temp', 'BUN', 'Alkalinephos', 'Creatinine']
    df = stat_measure(df, columns)

    # missing ratio of each column
    # missing_ratio[file_no, :] = 1.0 - df.count()/values.shape[0]

    # impute missing value by forward and backward filling in
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # impute missing value by column mean
    df = df.fillna(value=column_mean_dict)

    # construct the feature
    df = construct_feature(df)
    return df


def preprocess_data():
    # split the dataset
    data_dir = 'D:/Projects/physionet-master/training_data/training_stage1/'
    files = glob.glob(data_dir + '*.psv')
    patients = np.array(files)
    patients_no = len(files)        # 45336
    patients_train = 35336
    patients_test = 10000
    record_no = 1740663
    record_train = 1357085
    record_test = record_no - record_train
    columns_num = 40

    # init variables
    column_names = read_column_name(data_dir + 'p00001.psv')
    combined_values = np.array([], dtype=np.float64).reshape(0, columns_num+1)     # +1 denotes the label
    combined_imputed_df = pd.DataFrame(combined_values, columns=column_names)
    missing_ratio = np.zeros((len(files), columns_num+1)) - 1
    sepsis = np.zeros(len(files)) - 1
    file_no = 0
    record_no = 0
    df_list = []

    for file_name in files:
        if file_name.endswith('.psv'):
            (values, column_names) = read_raw_data(file_name)
            df = pd.DataFrame(values, columns=column_names)
            # count patient with sepsis
            sepsis[file_no] = df['SepsisLabel'].max()

            # stat the measure of important sign
            columns = ['WBC', 'Platelets', 'Temp', 'BUN', 'Alkalinephos', 'Creatinine']
            df = stat_measure(df, columns)

            # missing ratio of each column
            # missing_ratio[file_no, :] = 1.0 - df.count()/values.shape[0]

            # impute missing value by forward and backward filling in
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

            # construct the feature
            df = construct_feature(df)

            df_list.append(df)
            # show progress
            record_no = record_no + df.shape[0]
            # combined_values = np.vstack([combined_values, values]) if combined_values.size else values
            file_no += 1
            if file_no == patients_train:
                print(file_name)
                print('Current record:', record_no)                    # 1357085

            if file_no % 1000 == 0:
                print('Reading file: ', file_no)

    print(file_name)
    print(file_name)
    print('Current record:', record_no)                                 # 1740663
    print('Total files with sepsis:', int(np.sum(sepsis)))              # 3211
    print('Train files with sepsis:', np.sum(sepsis[:patients_train]))  # 2619
    print('Test files with sepsis:', np.sum(sepsis[patients_train:]))   # 592

    # combine the data
    print('Combining files...')
    t = time.time()
    df = pd.concat(df_list)
    print('Elapsed time:', time.time()-t)

    column_names = df.columns.values
    # column_mean_dict = read_pickle('column_mean.pickle')
    # df.fillna(column_mean_dict, inplace=True)
    # df.to_csv('D:/Projects/physionet-master/training_data/augmented_raw_stage1_data.csv', index=False)
    # df.to_csv('D:/Projects/physionet-master/training_data/raw_stage1_data.csv', index=False)

    # impute the data by mean
    imputed_df, meaner = impute_by_mean(df[column_names], column_names)
    imputed_df.to_csv('D:/Projects/physionet-master/training_data/aug_imputed_stage1_data.csv', index=False)    # set index to False

    # scale the data
    # imputed_scaled_df, scaler = scale_data(imputed_df[column_names], column_names)
    # imputed_scaled_df.to_csv('D:/Projects/physionet-master/training_data/aug_imputed_scaled_stage1_data.csv', index=False)

    # save scaler
    #scaler_filename = "aug_maxmin_scaler.save"
    # save_scaler(scaler, scaler_filename)

    # save column mean
    meaner = pd.DataFrame(meaner).T     # convert series to dataframe
    meaner.to_csv('D:/Projects/physionet-master/aug_column_mean.csv', index=False)
    print(meaner)
    meaner_dict = meaner.to_dict()
    filename = "aug_column_mean.pickle"
    to_pickle(meaner_dict, filename)


if __name__ == '__main__':
    preprocess_data()
