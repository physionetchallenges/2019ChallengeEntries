import pandas as pd
import numpy as np
from preprocess_data import *

data_path = 'D:/Projects/physionet-master/training_data/'
file_name = data_path + 'aug3_imputed_stage1_data.csv'
processed_df = pd.read_csv(file_name, engine='c')
eps = 1e-6
processed_df.loc[:, 'iculos_admin_ratio'] = processed_df.loc[:, 'iculos_max'] / (-processed_df.loc[:, 'HospAdmTime'] + eps)       #
processed_df.to_csv('aug2_imputed_stage1_data.csv', index=False)

print(processed_df['iculos_admin_ratio'].mean())

n_record = 1740663
n_train = 1357085  # 1357085
n_test = n_record - n_train
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(file_name, n_record, n_train, n_test)

a = np.argwhere(np.isnan(X_train))
b = np.argwhere(np.isnan(X_valid))
c = np.argwhere(np.isnan(X_test))
d = np.argwhere(np.isinf(X_train))
e = np.argwhere(np.isinf(X_valid))
f = np.argwhere(np.isinf(X_test))