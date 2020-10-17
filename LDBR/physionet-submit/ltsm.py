
import pandas as pd
import psd
import tensorflow as tf


cols_wecare = ["HR", "O2Sat","Temp", "SBP","MAP", "Resp", "ICULOS", "Gender", "Age"]

time_step = 6


def get_model():
    model = tf.keras.models.load_model("model1.sav")
    # summarize model.
    model.summary()
    return model


def get_features(data):
    # add header.
    pdfr = pd.DataFrame(data, columns=psd.c_header)

    engg_fl1 = psd.minmax_normalization(cols_wecare, pdfr, True)
    # engg_fl1 = psd.factor_ofnormal( cols_wecare, pdfr )
    engg_fl2 = psd.create_crossfeaturesR2(engg_fl1, [])
    engg_fl = engg_fl2
    features = psd.df_to_timestep_predict(engg_fl, time_step)

    return features


