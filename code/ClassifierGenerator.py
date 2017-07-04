from log.Print import *
from sklearn.svm import OneClassSVM
from keras.models import Model
from keras.layers import Input, Dense


def generate_one_class_svm_linear():
    svm = OneClassSVM(nu=0.5, kernel='linear')
    return svm


def generate_one_class_svm_sigmoid():
    svm = OneClassSVM(nu=0.5, kernel='sigmoid')
    return svm


def generate_one_class_svm_poly():
    svm = OneClassSVM(nu=0.5, kernel='poly')
    return svm


def generate_one_class_svm_rbf():
    svm = OneClassSVM(nu=0.5, kernel='rbf')
    return svm


def generate_autoencoder(input_shape, hidden_layer_size, input_size):
    input_img = Input(shape=input_shape)
    encoded = Dense(units=hidden_layer_size, input_shape=input_shape, activation='relu')(input_img)
    decoded = Dense(units=input_size, activation='hard_sigmoid')(encoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return encoder, autoencoder
