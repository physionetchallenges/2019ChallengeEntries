import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense, Masking, BatchNormalization


def weighted_crossentropy_loss(weight=0.7):
    def weighted_binary_crossentropy(y_label, y_pred):
        # loss for positiv label:
        tp_fp = - y_label * tf.log(y_pred + K.epsilon())

        # loss for negativ label:
        tn_fn = - (1 - y_label) * weight * tf.log(1 - y_pred + K.epsilon())

        return tp_fp + tn_fn

    def loss(y_label, y_pred):
        return weighted_binary_crossentropy(y_label[:, :, 0], y_pred[:, :, 1])

    return loss


# get the lstm as a keras model
def get_lstm(input_size, num_classes: int = 2):
    '''
    generates a compiles model of an lstm network

    :param input_size: the number of features the data will have (integer)
    :param lstm_size: how many hidden units the lstm layers will have
    :param num_classes: How many classes there are for the output
    :return: the compiled keras model
    '''

    lstm_units = [400, 400]
    dense_units = [250, 150, 100, 50]

    recurrent_dropout = 0.5
    dropout = 0.2

    input_shape = (None, input_size)
    inp = Input(input_shape)

    layer = inp
    masking = Masking(0)
    layer = masking(layer)
    layer = BatchNormalization()(layer)

    for l_size in lstm_units:
        layer = LSTM(l_size, return_sequences=True, implementation=2, dropout=dropout,
                     recurrent_dropout=recurrent_dropout)(layer)
        layer = BatchNormalization()(layer)

    for idx, d_size in enumerate(dense_units):
        layer = TimeDistributed(Dense(d_size, activation='relu'))(layer)
        layer = BatchNormalization()(layer)

    out = TimeDistributed(Dense(num_classes, activation="softmax"))(layer)

    model = keras.models.Model(inputs=inp, outputs=out)
    loss = weighted_crossentropy_loss()
    optimizer = keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])

    return model
