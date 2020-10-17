from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
# def get_data_from_file(inputFile):
# 	data = pd.read_csv(inputFile, delimiter='|', header=0)
# 	data = data.interpolate()
# 	data = data.fillna(method ='ffill')
# 	data = data.fillna(data.mean())
# 	data = data.fillna(0)

# 	return data


def prepare_test_data(data, mean_features, scaler):
	features = d_to_features3(data, mean_features, scaler)
	return features


def d_to_features3(training_data, mean_features, scaler):
	i = training_data.shape[0] 

	if i >=6:
		d = training_data[(i-6):i].flatten()
           
	else:
		d = training_data[0:i]
		imputing_values = mean_features[0:(len(mean_features)-1)].reshape(1,40)
		#print(imputing_values.shape)
		while(d.shape[0] < 6):
			d = np.concatenate([imputing_values, d], axis=0)
		d = d.flatten()
	#normalize features
	d = d.reshape((1,240))
	d = scaler.transform(d)
	d = d.reshape((240,))
	return d


# impute missing data
def impute_missing_data(data, mean_features):
	
	df = pd.DataFrame(data)
	for idx, column_name in enumerate(df.columns):
		replace_value = mean_features[idx]
		df[column_name] = df[column_name].fillna(value=replace_value)
	return df.values

