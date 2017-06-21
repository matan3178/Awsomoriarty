import definitions
from code.FetureExtractorUtil import start, samples_to_np_arrays
from log.Print import *
from code.FileLoader import FileLoader
from code.ClassifierGenerator import *


def do_something():
    fl = FileLoader(use_cache=False)
    hashes, training, testing=fl.load_collection2v3(definitions.COLLECTION3V6_DIR)
    print("extracting features...", HEADER)
    training_set = start(training[hashes[0]][:18000])
    testing_set = start(training[hashes[1]][:18000])
    print("generating svm...", HEADER)
    svm = generateOneClassSVM_Mashu()
    print("training... ({} samples)".format(len(training_set)), HEADER)
    svm.fit(training_set)
    print("predicting...", HEADER)
    print(len([y for y in svm.predict(testing_set) if y == 1]))

    return


def main():
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)
    return

if __name__ == "__main__":
    main()
