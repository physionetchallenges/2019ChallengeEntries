
# sequency_processing.py

import pandas as pd
import numpy as np

######################
###### CLEANING ######
######################

def generate_limits_vitals():
    """ Generate plausible ranges for the vital signs  
    """
    indx = [ 'HR', 'Resp', 'O2Sat', 'SBP', 'DBP', 'Temp', 'MAP'] 
    lims = [ [20, 300],[1,80],[20,100],[10,300],[10, 280],[28,48],[20, 300] ]
    LIMS = pd.DataFrame(lims, columns=['LOWER', 'UPPER'], index=indx)
    return LIMS

def generate_limits_labs():
    """ Generate plausible ranges for the lab results
    """
    indx = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
       'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
       'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
       'WBC', 'Fibrinogen']
    LIMS = pd.DataFrame(index=indx,columns=['LOWER', 'UPPER'])
    return LIMS
     
def clean_vitals(df):
    """ Clean data frame by removing implausible or nonphysiologic values for vital signs
    """
    LIMS = generate_limits_vitals()
    
    for vs in LIMS.index:
        df.loc[(df[vs] < LIMS.loc[vs, 'LOWER']) | 
               (df[vs] > LIMS.loc[vs, 'UPPER']), vs] = np.nan
    
    return df


######################
##### PROCESSING #####
######################

def impute_missing_data(X):
    """ impute median values found for each feature of input data
    """
    # average for all the features included in the model
    x_mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777,
        60.8711, 1.0, 0.0, 0.0, -59.6769,
        0.68024, 2.8557, 
        0.0, 0.0, 0.0, 0.0,
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280,
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280, 97.0520, 
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280,
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280, 97.0520,
        28.0])
    
    #Find indicies that you need to replace
    inds = np.where(np.isnan(X))
    
    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(x_mean, inds[1])
    
    return X

def _missing_variable_timing(time, values):
    """ determine cumulative / elapsed time since last non-null value
    """
    missing_flag = ~np.isnan(values)
    time_1 = np.maximum.accumulate(missing_flag*time)
    time_2 = time - time_1
    time_2 = np.array(time_2, dtype=float)
    time_2[time_1==0] = np.nan
    
    return time_2

def missing_flag_data(df, FEAT_LIST):
    """ calculate time since last (non-null) recorded value
    """
    for FEAT in FEAT_LIST:
        df[FEAT + '_MissFlag'] = _missing_variable_timing(df['ICULOS'].values, df[FEAT].values)
    
    return df

##### no need
'''
def _missing_variable_timing(time, values):
    """ Calculate time since last measured variable
    """
    non_missing_flag = ~np.isnan(values)
    if np.sum(non_missing_flag) == 0:
        return np.nan
    
    return time[-1] - time[non_missing_flag==True][-1]
    
def missing_flag(df, FEAT_LIST):
    for FEAT in FEAT_LIST:
        df[FEAT + '_MissFlag'] = _missing_variable_timing(df['ICULOS'].values, df[FEAT].values)
    
    return df
'''

def maximum_over_window(df, FEAT_LIST, WIN=12):
    """ find last WIN hours, and compute maximum value for each one of the features over that period
    """
    for FEAT in FEAT_LIST:
        df[FEAT + '_MAX' + str(WIN)] = np.max(df[FEAT].iloc[-WIN:])
    
    return df
    
def augmentMaximumOverWindow(df, FEAT_LIST, WIN=12):
    #df_ = df.copy()
    #out_list = []
    for FEAT in FEAT_LIST:
        name_ = FEAT + '_MAX' + str(WIN)
        df[name_] = df[FEAT].rolling(WIN, min_periods=1).max()
        #out_list.append(name_)
    return df

def minimum_over_window(df, FEAT_LIST, WIN=12):
    """  find last WIN hours, and compute minimum value for each one of the features over that period
    """ 
    for FEAT in FEAT_LIST:
        df[FEAT + '_MIN' + str(WIN)] = np.min(df[FEAT].iloc[-WIN:])
    
    return df
    
def augmentMinimumOverWindow(df, FEAT_LIST, WIN=12):
    #df_ = df.copy()
    #out_list = []
    for FEAT in FEAT_LIST:
        name_ = FEAT + '_MIN' + str(WIN)
        df[name_] = df[FEAT].rolling(WIN, min_periods=1).min()
        #out_list.append(name_)
    return df

def change_over_window(df, FEAT_LIST):
    for FEAT in FEAT_LIST:
        df[FEAT + '_CHANGE'] = _change_values(df[FEAT].values)
    
    return df
    
def augmentChangeOverWindow(df, FEAT_LIST, WIN=120):
    #df_ = df.copy()
    #out_list = []
    for FEAT in FEAT_LIST:
        name_ = FEAT + '_CHANGE'
        df[name_] = df[FEAT].rolling(WIN, min_periods=1).apply(_changeValues, raw=False)
        #out_list.append(name_)
    return df

def augment_interactions(data):
    """ compute interaction terms between different variables
    (non-linear associations)
    """
    #data = data_.copy()
    # Derived MAP from systolic and diastolic (if needed)
    data['MAPc'] = 1*data['SBP']/3 + 2*data['DBP']/3
    # shock index
    data['SI'] = data['HR']/data['SBP']
    # BUN-to-creatinine ratio
    data['BUNCRLOG'] = np.log(data['BUN']/data['Creatinine'])
    
    return data

def fill_missing_data(data, method='ffill', features=['HR']):
    """ Fill in missing data using different strategies 

    : type method: string
    : param method: what method to be used to fill in missing data
        e.g., locf (last observation carried forward)
    """
    #datam = data.copy()
    data[features] = data[features].fillna(method='ffill')
    return data

def prepare_data(data, feature_labels, feature_map):
    """ prepare data: call all required functions for cleaning, processing and augmenting the input data set
    """
    # create data frame
    df = pd.DataFrame(data=data, columns=feature_labels)
    # clean data
    df = clean_vitals(df)
    # compute missing flag data (which includes timing since observation)
    df = missing_flag_data(df,feature_labels[:34])
    # calculate change in value for certain labs
    df = change_over_window(df, ['Creatinine', 'BUN', 'Bilirubin_total', 'Platelets'])
    # need to fill missing data
    df = fill_missing_data(df, features=feature_labels[:34])
    # augment data
    df = maximum_over_window(df, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp'], WIN=12)
    df = minimum_over_window(df, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp', 'O2Sat'], WIN=12)
    df = maximum_over_window(df, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp'], WIN=24)
    df = minimum_over_window(df, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp', 'O2Sat'], WIN=24)
    # set interactions
    df = augment_interactions(df)
    # impute missing data
    input_x = impute_missing_data(df[feature_map].values)
    
    return input_x

def _change_values(x):
    return np.nan if x[~np.isnan(x)].size < 2 else np.diff(x[~np.isnan(x)])[-1]

def news_fio2(x):
    score=0.0
    # FiO2
    if x[10] > .22:
        score =+ 2.0
    return score

def news_hr(x):
    score=0.0
    # HR
    if x[0] <= 40:
        score =+ 3.0
    elif x[0] <= 50:
        score =+ 1.0
    elif x[0] <= 90:
        score =+ 0.0
    elif x[0] <= 110:
        score =+ 1.0
    elif x[0] <= 130:
        score =+ 2.0
    elif x[0] <= 301:
        score =+ 3.0
    return score

def news_spo2(x):
    score=0.0
    # SPO2
    if x[1] <= 91:
        score =+ 3.0
    elif x[1] <= 93:
        score =+ 2.0
    elif x[1] <= 95:
        score =+ 1.0
    elif x[1] <= 101:
        score =+ 0.0
    return score

def news_temp(x):
    score=0.0
    # Temp
    if x[2] <= 35.0:
        score =+ 3.0
    elif x[2] <= 36.0:
        score =+ 1.0
    elif x[2] <= 38.0:
        score =+ 0.0
    elif x[2] <= 39.0:
        score =+ 1.0
    elif x[2] <= 45.0:
        score =+ 2.0
    return score

def news_sbp(x):
    score=0.0
    # SBP
    if x[3] <= 90:
        score =+ 3.0
    elif x[3] <= 100:
        score =+ 2.0
    elif x[3] <= 110:
        score =+ 1.0
    elif x[3] <= 219:
        score =+ 0.0
    elif x[3] <= 301:
        score =+ 3.0
    return score

def news_rr(x):
    score=0.0
    # RR
    if x[6] <= 8:
        score =+ 3.0
    elif x[6] <= 11:
        score =+ 1.0
    elif x[6] <= 20:
        score =+ 0.0
    elif x[6] <= 24:
        score =+ 2.0
    elif x[6] > 24:
        score =+ 3.0
    return score

def calculate_news(X):
    res = np.apply_along_axis(news_hr, 1, X) + np.apply_along_axis(news_sbp, 1, X) + np.apply_along_axis(news_temp, 1, X) + np.apply_along_axis(news_rr, 1, X) + np.apply_along_axis(news_spo2, 1, X) + np.apply_along_axis(news_fio2, 1, X)    
    return res