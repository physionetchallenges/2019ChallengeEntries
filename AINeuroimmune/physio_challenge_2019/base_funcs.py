import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from inference import my_kalman_smooth, my_kalman_pred, my_kalman_pred_2, my_ar, my_arma
from calc_ews import q_sofa, sofa, news, mews


def read_pdata (f):
    drp_col = ['Age', 'Gender', 'HospAdmTime', 'Unit1', 'Unit2', 'SepsisLabel']
    p_file = pd.read_csv(f, sep='|')
    p_file = p_file.fillna(method='ffill')
    p_file = p_file.fillna(method='bfill')
    p_diff1 = p_file.diff(periods=1).drop(columns=drp_col)
    p_diff1.rename(columns=lambda x: x+'_d1', inplace=True)
    p_diff1 = p_diff1.fillna(0)
    p_diff2 = p_file.diff(periods=2).drop(columns=drp_col)
    p_diff2.rename(columns=lambda x: x+'_d2', inplace=True)
    p_diff2 = p_diff2.fillna(0)
    p_diff3 = p_file.diff(periods=3).drop(columns=drp_col)
    p_diff3.rename(columns=lambda x: x+'_d3', inplace=True)
    p_diff3 = p_diff3.fillna(0)
    p_diff4 = p_file.diff(periods=4).drop(columns=drp_col)
    p_diff4.rename(columns=lambda x: x+'_d4', inplace=True)
    p_diff4 = p_diff4.fillna(0)
    p_mean5 = p_file.rolling(5).mean().drop(columns=drp_col)
    p_mean5.rename(columns=lambda x: x+'_avg', inplace=True)
    p_mean5.iloc[[0]] = p_file.iloc[[0]].drop(columns=drp_col)
    p_mean5.loc[1, :] = np.mean(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_mean5.loc[2, :] = np.mean(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_mean5.loc[3, :] = np.mean(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)
    p_std5 = p_file.rolling(5).std().drop(columns=drp_col)
    p_std5.rename(columns=lambda x: x+'_std', inplace=True)
    p_std5.iloc[[0]] = 0
    p_std5.loc[1, :] = np.std(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_std5.loc[2, :] = np.std(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_std5.loc[3, :] = np.std(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)
    p_min5 = p_file.rolling(5).min().drop(columns=drp_col)
    p_min5.rename(columns=lambda x: x+'_min', inplace=True)
    p_min5.iloc[[0]] = p_file.iloc[[0]].drop(columns=drp_col)
    p_min5.loc[1, :] = np.min(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_min5.loc[2, :] = np.min(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_min5.loc[3, :] = np.min(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)  
    p_max5 = p_file.rolling(5).max().drop(columns=drp_col)
    p_max5.rename(columns=lambda x: x+'_max', inplace=True)
    p_max5.iloc[[0]] = p_file.iloc[[0]].drop(columns=drp_col)
    p_max5.loc[1, :] = np.max(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_max5.loc[2, :] = np.max(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_max5.loc[3, :] = np.max(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)   
    p_file['filename'] = pd.Series([f[-10:-4] for x in range(len(p_file.index))])
    p_file2 = pd.concat([p_file, p_diff1, p_diff2, p_diff3, p_diff4, p_mean5, p_std5, p_min5, p_max5],
                        axis=1, sort=False)
    
    return p_file2


def read_fex_psv(data, cols):
    skip_kalman = True
    freq_kalm = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Glucose']
    drp_col = ['EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'HospAdmTime', 'Unit1', 'Unit2', 'Age', 'Gender', 'ICULOS']
    
    p_file = pd.DataFrame(data=data, columns=cols[:-1])
    p_file = p_file.fillna(method='ffill')
    p_file = p_file.fillna(method='bfill')
    p_diff1 = p_file.diff(periods=1).drop(columns=drp_col)
    p_diff1.rename(columns=lambda x: x + '_d1', inplace=True)
    p_diff1 = p_diff1.fillna(0)
    
    p_diff2 = p_file.diff(periods=2).drop(columns=drp_col)
    p_diff2.rename(columns=lambda x: x + '_d2', inplace=True)
    p_diff2 = p_diff2.fillna(0)
    
    p_diff3 = p_file.diff(periods=3).drop(columns=drp_col)
    p_diff3.rename(columns=lambda x: x + '_d3', inplace=True)
    p_diff3 = p_diff3.fillna(0)
    
    p_diff4 = p_file.diff(periods=4).drop(columns=drp_col)
    p_diff4.rename(columns=lambda x: x + '_d4', inplace=True)
    p_diff4 = p_diff4.fillna(0)
    
    p_diff5 = p_file.diff(periods=5).drop(columns=drp_col)
    p_diff5.rename(columns=lambda x: x + '_d5', inplace=True)
    p_diff5 = p_diff5.fillna(0)
    
    p_diff6 = p_file.diff(periods=6).drop(columns=drp_col)
    p_diff6.rename(columns=lambda x: x + '_d6', inplace=True)
    p_diff6 = p_diff6.fillna(0) 
    
    p_mean6 = p_file.rolling(6).mean().drop(columns=drp_col)
    p_mean6.rename(columns=lambda x: x + '_avg', inplace=True)
    p_mean6.iloc[[0]] = p_file.iloc[[0]].drop(columns=drp_col)
    p_mean6.loc[1, :] = np.mean(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_mean6.loc[2, :] = np.mean(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_mean6.loc[3, :] = np.mean(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)
    p_mean6.loc[4, :] = np.mean(p_file.loc[0:4].drop(columns=drp_col).values, axis=0)
    
    p_std6 = p_file.rolling(6).std().drop(columns=drp_col)
    p_std6.rename(columns=lambda x: x + '_std', inplace=True)
    p_std6.iloc[[0]] = 0
    p_std6.loc[1, :] = np.std(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_std6.loc[2, :] = np.std(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_std6.loc[3, :] = np.std(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)
    p_std6.loc[4, :] = np.std(p_file.loc[0:4].drop(columns=drp_col).values, axis=0)
    
    p_min6 = p_file.rolling(6).min().drop(columns=drp_col)
    p_min6.rename(columns=lambda x: x + '_min', inplace=True)
    p_min6.iloc[[0]] = p_file.iloc[[0]].drop(columns=drp_col)
    p_min6.loc[1, :] = np.min(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_min6.loc[2, :] = np.min(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_min6.loc[3, :] = np.min(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)
    p_min6.loc[4, :] = np.min(p_file.loc[0:4].drop(columns=drp_col).values, axis=0)
    
    p_max6 = p_file.rolling(6).max().drop(columns=drp_col)
    p_max6.rename(columns=lambda x: x + '_max', inplace=True)
    p_max6.iloc[[0]] = p_file.iloc[[0]].drop(columns=drp_col)
    p_max6.loc[1, :] = np.max(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_max6.loc[2, :] = np.max(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_max6.loc[3, :] = np.max(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)
    p_max6.loc[4, :] = np.max(p_file.loc[0:4].drop(columns=drp_col).values, axis=0)
    
    p_med6 = p_file.rolling(6).median().drop(columns=drp_col)
    p_med6.rename(columns=lambda x: x + '_med', inplace=True)
    p_med6.iloc[[0]] = p_file.iloc[[0]].drop(columns=drp_col)
    p_med6.loc[1, :] = np.median(p_file.loc[0:1].drop(columns=drp_col).values, axis=0)
    p_med6.loc[2, :] = np.median(p_file.loc[0:2].drop(columns=drp_col).values, axis=0)
    p_med6.loc[3, :] = np.median(p_file.loc[0:3].drop(columns=drp_col).values, axis=0)
    p_med6.loc[4, :] = np.median(p_file.loc[0:4].drop(columns=drp_col).values, axis=0)
    
    if not skip_kalman:
        p_kalm = p_file[freq_kalm].rolling(6).apply(lambda x: x[-1] - my_ar(x), raw=True)
        p_kalm.rename(columns=lambda x: x + '_kalm', inplace=True)   
        p_kalm = p_kalm.fillna(0)
    
    p_file['qsofa'] = p_file.apply (lambda row: q_sofa(np.nan ,row.Resp, row.SBP), axis=1)
    p_file['qsofa_d1']  = p_file['qsofa'].diff(periods=1)
    p_file['qsofa_d1'] = p_file['qsofa_d1'].fillna(0)
    p_file['sofa'] = p_file.apply (lambda row: sofa(row.FiO2 ,np.nan, False, row.Platelets, np.nan, row.Bilirubin_direct, row.Bilirubin_total, row.MAP, np.nan, np.nan, np.nan, np.nan, row.Creatinine), axis=1)
    p_file['sofa_d1']  = p_file['sofa'].diff(periods=1)
    p_file['sofa_d1'] = p_file['sofa_d1'].fillna(0)
    p_file['news'] = p_file.apply (lambda row: news(row.Resp , row.O2Sat, row.Temp, row.SBP, row.HR), axis=1)
    p_file['news_d1']  = p_file['news'].diff(periods=1)
    p_file['news_d1'] = p_file['news_d1'].fillna(0)
    p_file['mews'] = p_file.apply (lambda row: mews(row.Resp, row.Temp, row.SBP, row.HR), axis=1)
    p_file['mews_d1']  = p_file['mews'].diff(periods=1)
    p_file['mews_d1'] = p_file['mews_d1'].fillna(0)    
    p_file = p_file.fillna(p_file.mean())    
    
    if not skip_kalman:
        p_file2 = pd.concat([p_file, p_diff1, p_diff2, p_diff3, p_diff4, p_diff5,
                         p_diff6, p_mean6, p_std6, p_min6, p_max6, p_med6, p_kalm], axis=1, sort=False)
    else:
        p_file2 = pd.concat([p_file, p_diff1, p_diff2, p_diff3, p_diff4, p_diff5,
                         p_diff6, p_mean6, p_std6, p_min6, p_max6, p_med6], axis=1, sort=False)
    
    return p_file2


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = RobustScaler()
    scaler = StandardScaler()
    #scaler = Normalizer()	
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled