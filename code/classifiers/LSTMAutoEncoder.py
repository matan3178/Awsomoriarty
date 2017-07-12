import numpy as np
from code import _definitions
from code.features.FetureExtractorUtil import sliding_windows, flatten_windows
from code.log.Print import *
from code.utils import mse, flatten_list


class LSTMAutoEncoder:

    encoder_decoder = "UNINITIALIZED"
    threshold = "UNINITIALIZED"
    name = "UNINITIALIZED"
    epochs_number = "UNINITIALIZED"
    batch_size = "UNINITIALIZED"

    window_size = "UNINITIALIZED"
    last_window = "UNINITIALIZED"

    def __init__(self, inner_autoencoder, name="AutoEncoder (?)", threshold=0.5, epochs_number=1, batch_size=1, window_size=5):
        self.encoder_decoder = inner_autoencoder
        self.name = name
        self.threshold = threshold
        self.epochs_number = epochs_number
        self.batch_size = batch_size

        self.window_size = window_size
        return

    def get_name(self):
        return self.name

    def fit(self, training_set):
        x_train = sliding_windows(training_set, self.window_size, 1)
        self.last_window = x_train[len(x_train) - 1]
        x_train = np.array(x_train)

        y_train = flatten_windows(x_train)
        y_train = np.array(y_train)

        print("lstm input shape: {}; output shape: {}".format(x_train.shape[1:], y_train.shape[1:]), COMMENT)
        print("lstm training_set size: {}".format(x_train.shape[0]), COMMENT)

        self.encoder_decoder.fit(x_train, y_train, epochs=self.epochs_number, batch_size=self.batch_size,
                                 verbose=_definitions.VERBOSITY_training_autoencoder, shuffle=True)
        return

    def predict(self, x_test):
        x_test = sliding_windows(x_test, self.window_size, 1)

        reconstructions = self.encoder_decoder.predict(np.array(x_test))
        predictions = list()
        # the first [window_size - 1] samples fill the window,
        # the sample at index [window_size] is the first to generate a real prediction
        #
        # generate dummy prediction for first [window_size - 1] samples
        # |
        # v
        predictions.extend(np.zeros(self.window_size - 1))
        for reconst, x in zip(reconstructions, x_test):
            if mse(reconst, flatten_list(x)) > self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        print("prediction successful!", HEADER)
        return predictions

    def predict_single(self, sample):
        if len(self.last_window) < self.window_size:
            self.last_window.append(sample)
            return 0

        self.last_window = self.last_window[1:]
        self.last_window.append(sample)

        reconstruction = self.encoder_decoder.predict(np.array([self.last_window]))
        # print(reconstruction.shape, OKBLUE)
        return 1 if mse(reconstruction[0], flatten_list(self.last_window)) > self.threshold else 0

    def has_threshold(self):
        return True

    def set_threshold(self, threshold):
        self.threshold = threshold
        return
