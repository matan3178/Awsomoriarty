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

    def predict_mses(self, x_test):
        x_test = np.array(x_test)
        reconstructions = self.encoder_decoder.predict(x_test)
        mses = [mse(reconst, x) for reconst, x in zip(reconstructions, x_test)]
        return mses

    def get_predictions_from_mses(self, mses):
        return [1 if mse > self.threshold else 0 for mse in mses]

    def predict(self, x_test):
        return self.get_predictions_from_mses(self.predict_raw(x_test))

    def predict_single(self, sample):
        return 1 if self.predict(list([sample]))[0] > self.threshold else 0

    def has_threshold(self):
        return True

    def set_threshold(self, threshold):
        self.threshold = threshold
        return

    def supports_repredictions(self):
        return True

    def predict_raw(self, x_test):
        return self.predict_mses(x_test)

    def repredict(self, raw_data):
        return self.get_predictions_from_mses(raw_data)
