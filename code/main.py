import definitions
from code.FetureExtractorUtil import start, samples_to_np_arrays
from log.Print import *
from code.FileLoader import FileLoader
from code.ClassifierGenerator import *


def do_something():
    fl = FileLoader(use_cache=False)
    hashes, training, testing=fl.load_collection3v2(definitions.COLLECTION3V2_DIR)
    print("found {} samples in training file of user {}".format(len(training[hashes[0]]), hashes[0]))
    print("extracting features...", HEADER)
    training_set = start(training[hashes[0]][:18000])
    testing_set = start(training[hashes[1]][:18000])
    print("generating svm...", HEADER)
    svm = generate_one_class_svm_specific()
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
