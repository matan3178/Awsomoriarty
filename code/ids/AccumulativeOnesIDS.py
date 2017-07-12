class AccumulativeOnesIDS:
    classifier = "UNINITIALIZED"
    threshold = "UNINITIALIZED"
    num_of_contiguous_ones = 0

    def __init__(self, classifier, threshold=1):
        self.classifier = classifier
        self.threshold = threshold
        return

    def get_name(self):
        return "{}_AccumulativeOnes({})".format(self.threshold, self.classifier.get_name())

    def alert_if_theft(self, sample):
        pred = self.classifier.predict_single(sample)
        if pred == 1:
            self.num_of_contiguous_ones += 1
        else:
            self.num_of_contiguous_ones -= 1
            if self.num_of_contiguous_ones < 0:
                self.num_of_contiguous_ones = 0

        if self.num_of_contiguous_ones > self.threshold:
            return 1
        return 0

    def has_threshold(self):
        return True

    def set_threshold(self, threshold):
        self.threshold = threshold
        return
