#!/usr/bin/env python

import numpy as np
import pandas as pd
import  math
from joblib import  load

def get_sepsis_score(data, model):
    means_data = {
		"HR": 90.144276,
		"O2Sat": 97.167208,
		"Temp": 36.948197,
		"SBP": 127.364370,
		"MAP": 87.843218,
		"DBP": 67.597138,
		"Resp": 19.834761,
		"FiO2": 0.460233,
		"AST": 110.495045,
		"BUN": 24.958870,
		"Alkalinephos": 77.317032,
		"Calcium": 7.283898,
		"Creatinine"    :        1.611880,
		"Bilirubin_direct": 0.397330,
		"Glucose": 140.049961,
		"Lactate": 1.879757,
		"Magnesium": 2.077681,
		"Phosphate": 3.289671,
		"Potassium": 4.021308,
		"Hct": 31.350837,
		"Hgb": 10.216015,
		"PTT": 38.628155,
		"WBC": 12.857952,
		"Platelets": 205.103028
	}
    X = []
    k = 0
    time_step = 5

    column_names = [
		"HR","O2Sat","Temp","SBP","MAP","DBP","Resp",
		"EtCO2","BaseExcess","HCO3","FiO2","pH","PaCO2",
		"SaO2","AST","BUN","Alkalinephos","Calcium","Chloride",
		"Creatinine","Bilirubin_direct","Glucose","Lactate","Magnesium",
		"Phosphate","Potassium","Bilirubin_total","TroponinI","Hct","Hgb",
		"PTT","WBC","Fibrinogen","Platelets","Age","Gender","Unit1","Unit2",
		"HospAdmTime","ICULOS"
	]
    df = pd.DataFrame(data=data,  # values
    columns = column_names )

    sLength = len(df['HR'])
    # ____deleted select "O2Sat","SBP","MAP","DBP","AST","Bilirubin_direct","PTT","Age"
    df = df.drop(["Unit1", "Unit2", "HospAdmTime", 'ICULOS', "BaseExcess", "HCO3", "pH", "EtCO2",
                  "PaCO2", "SaO2", "Bilirubin_total", "Chloride", "TroponinI", "Fibrinogen"], axis=1)

    first_valid_val = df.apply(lambda col: col.first_valid_index()).to_dict()
    for i, j in first_valid_val.items():
        t = first_valid_val[i]
        if not math.isnan(t):
            df.loc[0:int(j), i] = df.loc[0:int(j), i].fillna(df.iloc[int(t)][i])
    df = df.interpolate()


    columns = df.columns

    for i in range(sLength):
        mask = df.loc[i].isnull().values
        cols = columns[mask]
        for l in range(len(cols)):
            df.at[i, cols[l]] = means_data[cols[l]]



    Xtest = df.loc[sLength-1].values
    Xtest=Xtest.reshape(1,-1)

    y_pridect = model.predict(Xtest)

    if y_pridect == -1 or y_pridect > 50:
        label = 0
        prob = 0.0
    elif y_pridect == 0:
        label = 1
        prob = 100.0
    else:
        prob = 0.01+(y_pridect-50)*(-0.02)
        label = prob > 0.40

    #score = 1 - np.exp(-l_exp_bx)
    #label = score > 0.45

    return prob, label

def load_sepsis_model():
    model = load( 'classifier2Reg05212019.pkl')
    return model
