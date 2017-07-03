import glob
import os
from genericpath import isfile
from macpath import join

from log.Print import *
import csv


class FileLoader:

    use_cache = True
    cache = dict()

    def __init__(self, use_cache):
        self.use_cache = use_cache
        return

    def load_csv_file(self, path):
        if self.use_cache and path in self.cache:
            return self.cache[path]
        else:
            with open(path) as datafile:
                lines = enumerate(csv.reader(datafile, delimiter=","))
                data = [line[1] for line in lines]
                if self.use_cache:
                    self.cache[path] = data
                return data

    def load_collection3v2(self, collection_path):
        user_hashes = list()
        users_training = dict()
        users_testing = dict()
        for user_dir in os.listdir(collection_path):

            user_hash = user_dir
            user_hashes.append(user_hash)

            training_path = "{}/{}/{}_training.csv".format(collection_path, user_hash, user_hash)
            print("loading {}".format("{}/{}".format(collection_path, user_hash)))
            users_training[user_hash] = self.load_csv_file(training_path)
            users_testing[user_hash] = [self.load_csv_file(test_file) for test_file in os.listdir("{}/{}".format(collection_path, user_hash)) if isfile(join(collection_path, user_hash, test_file))]

        return user_hashes, users_training, users_testing
