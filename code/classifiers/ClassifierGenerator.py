from keras.layers import Input, Dense, LSTM
from keras.models import Model
from sklearn.svm import OneClassSVM
from code.log.Print import *
from code.classifiers.AutoEncoder import AutoEncoder
from code.classifiers.LSTMAutoEncoder import LSTMAutoEncoder
from code.classifiers.OneClassSVMCustomized import OneClassSVMCustomized


def generate_one_class_svm_linear():
    svm = OneClassSVM(nu=0.1, kernel='linear', gamma=0.1, verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Linear")


def generate_one_class_svm_sigmoid():
    svm = OneClassSVM(nu=0.1, kernel='sigmoid', gamma=0.1, verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Sigmoid")


def generate_one_class_svm_poly():
    svm = OneClassSVM(nu=0.1, kernel='poly', gamma=0.1, verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Poly")


def generate_one_class_svm_rbf():
    svm = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1, verbose=False)
    return OneClassSVMCustomized(svm, name="OneClassSVM_Rbf")


def generate_autoencoder(input_size):

    input_layer = Input(shape=(input_size,))
    hidden = Dense(units=int(input_size * 0.3), activation='sigmoid')(input_layer)
    output_layer = Dense(units=input_size, activation='linear')(hidden)

    # encoder = Model(input_img, encoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return AutoEncoder(inner_autoencoder=autoencoder,
                       name="AutoEncoder({})->({})->({})".format(input_size, int(input_size * 0.2), input_size),
                       epochs_number=20,
                       batch_size=2)


def generate_lstm_autoencoder(sample_size, window_size):
    input_layer = Input(shape=(window_size, sample_size))
    hidden = LSTM(units=window_size, activation='tanh',
                  recurrent_activation='hard_sigmoid',
                  use_bias=True,
                  kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal',
                  bias_initializer='zeros',
                  unit_forget_bias=True,
                  kernel_regularizer=None,
                  recurrent_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  recurrent_constraint=None,
                  bias_constraint=None,
                  dropout=0.0,
                  recurrent_dropout=0.0)(input_layer)
    output_layer = Dense(units=window_size * sample_size, activation='sigmoid')(hidden)

    # encoder = Model(input_img, encoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return LSTMAutoEncoder(inner_autoencoder=autoencoder,
                           name="LSTM_AutoEncoder({},{})->({})->({})".format(window_size,
                                                                             sample_size,
                                                                             window_size,
                                                                             window_size * sample_size),
                           epochs_number=30, batch_size=5, window_size=window_size, threshold=0.05)
