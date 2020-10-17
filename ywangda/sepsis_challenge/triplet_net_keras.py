import numpy as np
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import random
import matplotlib.patheffects as PathEffects
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import os
import pickle
import matplotlib.pyplot as plt
from itertools import permutations
import seaborn as sns
from keras.datasets import mnist
from sklearn.manifold import TSNE
from sklearn.svm import SVC
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ------------------------------------------------------------------------------------------------
# Plot t-SNE embeddings on raw data.
# ------------------------------------------------------------------------------------------------
# Define our own plot function
def save_scatter_fig(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    if subtitle != None:
        plt.suptitle(subtitle)

    plt.savefig(subtitle)


x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)
tsne = TSNE()
train_tsne_embeds = tsne.fit_transform(x_train_flat[:512])
save_scatter_fig(train_tsne_embeds, y_train[:512], "Samples from Training Data")
eval_tsne_embeds = tsne.fit_transform(x_test_flat[:512])
save_scatter_fig(eval_tsne_embeds, y_test[:512], "Samples from Validation Data")


# ------------------------------------------------------------------------------------------------
# Train the pretrained model (MLP).
# ------------------------------------------------------------------------------------------------
Classifier_input = Input((784,))
Classifier_output = Dense(10, activation='softmax')(Classifier_input)
Classifier_model = Model(Classifier_input, Classifier_output)


from sklearn.preprocessing import LabelBinarizer
le = LabelBinarizer()
y_train_onehot = le.fit_transform(y_train)
y_test_onehot = le.transform(y_test)
Classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Classifier_model.fit(x_train_flat, y_train_onehot, validation_data=(x_test_flat, y_test_onehot), epochs=10)


# ------------------------------------------------------------------------------------------------
# Generate triplet embeddings on raw data.
# ------------------------------------------------------------------------------------------------
def generate_triplet(x, y, testsize=0.3, ap_pairs=10, an_pairs=10):
    print('generate triplets...')
    data_xy = tuple([x, y])

    trainsize = 1 - testsize

    triplet_train_pairs = []
    triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):       # get class labels
        print('class label:', data_class)
        same_class_idx = np.where((data_xy[1] == data_class))[0]             #
        diff_class_idx = np.where(data_xy[1] != data_class)[0]               #
        anchor_pos_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # get all permutations of length 2, sample k anchor-positive pairs
        neg = random.sample(list(diff_class_idx), k=an_pairs)      # sample k negative

        # triplets for training
        print('generate triplets for training...')
        A_P_len = len(anchor_pos_pairs)
        Neg_len = len(neg)
        for ap in anchor_pos_pairs[:int(A_P_len * trainsize)]:
            anchor = data_xy[0][ap[0]]
            positive = data_xy[0][ap[1]]
            for n in neg:
                negative = data_xy[0][n]
                triplet_train_pairs.append([anchor, positive, negative])

        # triplets for testing
        print('generate triplets for testing...')
        for ap in anchor_pos_pairs[int(A_P_len * trainsize):]:
            anchor = data_xy[0][ap[0]]
            positive = data_xy[0][ap[1]]
            for n in neg:
                negative = data_xy[0][n]
                triplet_test_pairs.append([anchor, positive, negative])
    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)

X_train, X_test = generate_triplet(x_train_flat,y_train, ap_pairs=150, an_pairs=150,testsize=0.2)


# ------------------------------------------------------------------------------------------------
# Train triplet NN.
# ------------------------------------------------------------------------------------------------

def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv2D(128, (7, 7), padding='same', input_shape=(in_dims[0], in_dims[1], in_dims[2],), activation='relu',
                     name='conv1'))
    model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='pool1'))
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(4, name='embeddings'))
    # model.add(Dense(600))
    return model


adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
anchor_input = Input((28,28,1,), name='anchor_input')
positive_input = Input((28,28,1,), name='positive_input')
negative_input = Input((28,28,1,), name='negative_input')

# Shared embedding layer for positive and negative items
print('create base network...')
shared_dnn = create_base_network([28, 28, 1,])
encoded_anchor = shared_dnn(anchor_input)
encoded_positive = shared_dnn(positive_input)
encoded_negative = shared_dnn(negative_input)
merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
model.compile(loss=triplet_loss, optimizer=adam_optim)
model.summary()

# reshape the flat feature to 2D
anchor = X_train[:, 0, :].reshape(-1,28,28,1)
positive = X_train[:, 1, :].reshape(-1,28,28,1)
negative = X_train[:, 2, :].reshape(-1,28,28,1)
anchor_test = X_test[:, 0, :].reshape(-1,28,28,1)
positive_test = X_test[:, 1, :].reshape(-1,28,28,1)
negative_test = X_test[:, 2, :].reshape(-1,28,28,1)

Y_dummy = np.empty((anchor.shape[0], 300))
Y_dummy2 = np.empty((anchor_test.shape[0], 1))

print('train TNN for embedding...')
model.fit([anchor,positive,negative],y=Y_dummy,validation_data=([anchor_test,positive_test,negative_test],Y_dummy2), batch_size=512, epochs=50, verbose=2)
# serialize weights to HDF5
print("save TNN model to disk...")
model.save_weights("triplet_model_MNIST.h5")


trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
trained_model.load_weights('triplet_model_MNIST.h5')

# draw scatter figure
tsne = TSNE()
X_train_trm = trained_model.predict(x_train[:512].reshape(-1,28,28,1))
X_test_trm = trained_model.predict(x_test[:512].reshape(-1,28,28,1))
train_tsne_embeds = tsne.fit_transform(X_train_trm)
eval_tsne_embeds = tsne.fit_transform(X_test_trm)
save_scatter_fig(train_tsne_embeds, y_train[:512], "Training Data After TNN")
save_scatter_fig(eval_tsne_embeds, y_test[:512], "Validation Data After TNN")

# predict
print('train TNN for classifying...')
X_train_trm = trained_model.predict(x_train.reshape(-1,28,28,1))
X_test_trm = trained_model.predict(x_test.reshape(-1,28,28,1))
Classifier_input = Input((4,))
Classifier_output = Dense(10, activation='softmax')(Classifier_input)
Classifier_model = Model(Classifier_input, Classifier_output)
Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
Classifier_model.fit(X_train_trm, y_train_onehot, validation_data=(X_test_trm, y_test_onehot), epochs=10, verbose=2)
# serialize weights to HDF5
print("save classifier model to disk...")
Classifier_model.save_weights("classifier_model_MNIST.h5")





