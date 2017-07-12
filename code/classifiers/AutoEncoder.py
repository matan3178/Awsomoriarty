import numpy as np

from code import _definitions
from code.utils import mse


class AutoEncoder:

    encoder_decoder = "UNINITIALIZED"
    threshold = "UNINITIALIZED"
    name = "UNINITIALIZED"
    epochs_number = "UNINITIALIZED"
    batch_size = "UNINITIALIZED"

    def __init__(self, inner_autoencoder, name="AutoEncoder (?)", threshold=0.5, epochs_number=1, batch_size=1):
        self.encoder_decoder = inner_autoencoder
        self.name = name
        self.threshold = threshold
        self.epochs_number = epochs_number
        self.batch_size = batch_size
        return

    def get_name(self):
        return "{} (threshold={})".format(self.name, self.threshold)

    def fit(self, x_train):
        x_train = np.array(x_train)
        self.encoder_decoder.fit(x_train, x_train, epochs=self.epochs_number, batch_size=self.batch_size,
                                 verbose=_definitions.VERBOSITY_training_autoencoder, shuffle=True)
        return

    def predict(self, x_test):
        x_test = np.array(x_test)
        reconstructions = self.encoder_decoder.predict(x_test)
        predictions = list()

        for reconst, x in zip(reconstructions, x_test):
            if mse(reconst, x) > self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def predict_single(self, sample):
        return 1 if self.predict(list([sample]))[0] > self.threshold else 0

    def has_threshold(self):
        return True

    def set_threshold(self, threshold):
        self.threshold = threshold
        return