# GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Pick which GPU to use "0-7"
# General
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
# Keras
import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Hyperparameters
batch_size = 1024
num_epoch = 1000
class_weight = {0:1, 1:10} # TODO: Needs calculation


def load_data(train_file, test_file):

    '''
    Input: npz files (files with store training and validation set)
    Returns: X_train with dimension (No. of training data, window_size, feature_size, 1)
             y_train with dimension (No. of training data, 1)
             X_test with dimension (No. of validation data, window_size, feature_size, 1)
             y_test with dimension (No. of validation data, 1)

    '''

    # Training set
    npz_file = np.load(train_file)
    X_train = npz_file['train_features']
    y_train = npz_file['train_outcomes']
    # Validation set
    npz_file = np.load(test_file)
    X_test = npz_file['valid_features']
    y_test = npz_file['valid_outcomes']

    window_size = X_train.shape[1] # TODO: Change to args.window_size?
    feature_size = X_train.shape[2] # TODO: Change to feature_length

    # Reshaping data for keras model
    X_train = X_train.reshape(X_train.shape[0], window_size, feature_size, 1)
    X_test = X_test.reshape(X_test.shape[0], window_size, feature_size, 1)
    input_shape = (window_size, feature_size, 1)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, y_train, X_test, y_test, input_shape

def build_cnn(input_shape):

    '''
    Build the structure of CNN model
    Input: input_shape: (window_size, feature_size, 1)
    Returns: model ready for training process
    '''

    # Build keras CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    print(model.summary())

    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size, num_epoch, class_weight, model_file):

    '''
    Model training
    Input: Data: X_train, y_train, X_test, y_test
           Hyperparameters: batch_size, num_epoch, class weight

    Returns: Trained and saved model
    '''
    # Early stopping
    callbacks = [EarlyStopping(monitor = 'val_acc', patience = 20),
                 ModelCheckpoint(model_file, save_best_only = True, monitor = 'val_acc')]

    #model training
    model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=num_epoch,
            callbacks = callbacks,
            class_weight = class_weight
            )
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Vaidation loss:', score[0])
    print('Vaidation accuracy:', score[1])


    return model

def save_results(model_file, npz_file, output_dir):

    '''
    Print simple evaluation results of the model and save predictions in physionet format
    Input: model_file, test_data file, output_dir
    '''

    # Load model
    model = load_model(model_file)
    npz = np.load(npz_file)
    X_test = npz['valid_features']
    y_test = npz['valid_outcomes']

    window_size = X_test.shape[1] # TODO: Change to args.window_size?
    feature_size = X_test.shape[2] # TODO: Change to feature_length


    X_test = X_test.reshape(X_test.shape[0], window_size, feature_size, 1)

    # Simple evaluation?
    print("Start prediction resuls ...")
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(y_pred.shape[0],)
    y_pred_class = []
    threshold = 0.25
    for i in y_pred:
        if i >= threshold:
            y_pred_class.append(1)
        else:
            y_pred_class.append(0)

    print("Classification report:\n", classification_report(y_test, y_pred_class))

    # Save to psv in output_dir
    filenames = [str(f)[2:-1] + "v" for f in npz_file["valid_filenames"]]
    results = pd.DataFrame({"filename": filenames,
                            "PredictedProbability": y_pred,
                            "PredictedLabel": y_pred_class,
                            "iculos": npz_file["valid_iculos"].reshape(npz_file["valid_iculos"].shape[0],)})

    os.chdir(output_dir)
    for filename_ in results.filename.unique():
        df = results[results.filename == filename_]["PredictedProbability", "PredictedLabel"]
        df.to_csv(filename_, index = False, sep = "|")

    return

if __name__ == "__main__":

    # Train CNN model from intermediate data files (.npz)
    '''
    X_train, y_train, X_test, y_test, input_shape = load_data(train_file, test_file)
    model = build_cnn(input_shape)
    model = train_model(model, X_train, y_train, X_test, y_test, batch_size, num_epoch, class_weight, model_file)
    save_results(model_file, test_file, output_dir)

    '''

    # Load exisiting model and save results
    output_dir = r"C:\Users\YangZh\Desktop\output_dir"
    train_file = r"Z:\LKS-CHART\Projects\physionet_sepsis_project\data\splits\split_0\train_preprocessed_measured_window_24.npz"
    test_file = r"Z:\LKS-CHART\Projects\physionet_sepsis_project\data\splits\split_0\valid_preprocessed_measured_window_24.npz"
    model_file = r"C:\Users\YangZh\Desktop\physionet-2019\cnn\physionet_cnn_ratio_0_1.h5"

    save_results(model_file, test_file, output_dir)
