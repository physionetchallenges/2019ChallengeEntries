import sys
import numpy as np

import sys
from xgboost import XGBClassifier
from sklearn.externals import joblib

from utils import *

from keras.layers import *
from keras.models import Model

from keras.optimizers import Adam
from keras.optimizers import SGD

# LSTM Autoencoder model
# this is the size of our encoded representations
encoding_dim = 64  

# this is our input placeholder
input_img = Input(shape=(6, 40))
x = BatchNormalization()(input_img)
encoded = LSTM(128, activation='relu', return_sequences=True)(x)
encoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
encoded = LSTM(encoding_dim, activation='relu')(encoded)

decoded = RepeatVector(6)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(40, activation='sigmoid'))(decoded)

classification_out = Dense(32, activation='relu')(encoded)
classification_out = Dropout(0.3)(classification_out)
classification_out = Dense(24, activation='relu')(classification_out)
classification_out = Dropout(0.3)(classification_out)
classification_out = Dense(16, activation='relu')(classification_out)
classification_out = Dropout(0.3)(classification_out)
classification_out = Dense(8, activation='relu')(classification_out)
classification_out = Dense(2, activation='softmax')(classification_out)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, [decoded, classification_out])
encoder = Model(input_img, encoded)
classifier = Model(input_img, classification_out)

adam = Adam(lr=0.001)
autoencoder.compile(optimizer=adam, loss=['mean_squared_error','categorical_crossentropy'],
                    loss_weights=[0.2, 0.8])

def load_sepsis_model():
    xgb = joblib.load('xgb_model')
    rf = joblib.load('rf_model')
    autoencoder.load_weights('encoder')

    return (xgb, rf, classifier)

def get_sepsis_score(data, model):
    #Load mean features
    mean_features = joblib.load('mean_features')
    mean_features = mean_features.values
    # Load scaler
    scaler = joblib.load('scaler')
    data = impute_missing_data(data, mean_features)
    features = prepare_test_data(data, mean_features, scaler)
    # Load models
    xgb = model[0]
    rf  = model[1]
    enc = model[2]

    #print(features.shape)
    xgb_output = xgb.predict_proba([features])
    xgb_probability = xgb_output[0][1]

    rf_output = rf.predict_proba([features])
    rf_probability = rf_output[0][1]

    enc_output = enc.predict([features.reshape((len([features]), 6, 40))])
    enc_probability = enc_output[0][1]
    
    score = max(xgb_probability, rf_probability, enc_probability)

    if score >0.5:
        label = 1
    else:
        label = 0
    return score, label
