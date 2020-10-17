import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input, MaxPooling2D, Dropout, Flatten
from keras import regularizers

from metric import *

weight_decay = 1e-4


def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


def vgg8(args):
    input = Input(shape=(28, 28, 1))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)

    return Model(input, output)


def vgg8_arcface(args):
    input = Input(shape=(28, 28, 1))
    y = Input(shape=(10,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = ArcFace(10, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)


def vgg8_cosface(args):
    input = Input(shape=(28, 28, 1))
    y = Input(shape=(10,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = CosFace(10, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)


def vgg8_sphereface(args):
    input = Input(shape=(28, 28, 1))
    y = Input(shape=(10,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = SphereFace(10, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)


def mlp_arcface(args):
    input_dim = 40
    pos_weight = 30
    ker_init = 'he_normal'
    batch_size = 16
    activation = 'relu'
    # hidden_dim = 64
    # hidden_dim2 = 8
    # drop_out = 0.1
    # prob_thresh = 0.5
    # run = 1

    input = Input(shape=(40,))
    y = Input(shape=(2,))

    x = Dense(200, input_dim=input_dim, kernel_initializer='he_normal')(input)           #
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.5)(x)

    x = Dense(100, input_dim=200, kernel_initializer='he_normal')(x)                     #
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.25)(x)

    x = Dense(50, input_dim=100, kernel_initializer='he_normal')(x)                      #
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.1)(x)

    x = Dense(25, input_dim=50, kernel_initializer='he_normal')(x)                       #
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.05)(x)

    # x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = ArcFace(2, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)