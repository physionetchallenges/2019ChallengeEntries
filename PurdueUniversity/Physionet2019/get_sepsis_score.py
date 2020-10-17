import numpy as np
import keras as K
from keras.models import model_from_json
from data_utils import clean_input

def get_sepsis_score(data, model):
    data = clean_input(data)
    data = data.reshape(1, data.shape[0], data.shape[1])
    score = model.predict(data)
    score = score[0][score.shape[1] - 1][0]
    label = score > 0.45
    return score, label

def load_sepsis_model():
    json_file = open('model.json', 'r')
    loaded_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_json)
    model.load_weights('model.h5')
    model.compile("adam", loss="binary_crossentropy", metrics=['binary_accuracy'])
    return model




