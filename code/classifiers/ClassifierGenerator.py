from code.classifiers.AutoEncoder import AutoEncoder
from code.classifiers.OneClassSVMCustomized import OneClassSVMCustomized
from log.Print import *
from sklearn.svm import OneClassSVM
from keras.models import Model
from keras.layers import Input, Dense


def generate_one_class_svm_linear():
    svm = OneClassSVM(nu=0.1, kernel='linear', verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Linear")


def generate_one_class_svm_sigmoid():
    svm = OneClassSVM(nu=0.1, kernel='sigmoid', verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Sigmoid")


def generate_one_class_svm_poly():
    svm = OneClassSVM(nu=0.1, kernel='poly', verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Poly")


def generate_one_class_svm_rbf():
    svm = OneClassSVM(nu=0.1, kernel='rbf', verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Rbf")


def generate_autoencoder(input_size):

    input_layer = Input(shape=(input_size,))
    hidden = Dense(units=int(input_size * 0.75), activation='relu')(input_layer)
    hidden = Dense(units=int(input_size * 0.5), activation='relu')(hidden)
    hidden = Dense(units=int(input_size * 0.75), activation='relu')(hidden)
    output_layer = Dense(units=input_size, activation='sigmoid')(hidden)

    # encoder = Model(input_img, encoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return AutoEncoder(inner_autoencoder=autoencoder,
                       name="AutoEncoder {}>{}>{}<{}<{}".format(input_size,
                                                                int(input_size * 0.75),
                                                                int(input_size * 0.5),
                                                                int(input_size * 0.75),
                                                                input_size),
                       epochs_number=50,
                       batch_size=5)
