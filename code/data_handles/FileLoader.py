import csv
import os
from genericpath import isfile
from code.log.Print import *


class FileLoader:

    use_cache = True
    cache = dict()

    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        return

    def load_csv_file(self, path, verbosity=0):
        if verbosity >= 2:
            print(path, COMMENT)

        if self.use_cache and path in self.cache:
            return self.cache[path]
        else:
            with open(path) as datafile:
                lines = enumerate(csv.reader(datafile, delimiter=","))
                data = [line[1] for line in lines]
                if self.use_cache:
                    self.cache[path] = data
                return data

    def load_collection3v2(self, collection_path, verbosity=0, num_of_users=-1, num_of_tests_per_user=-1):
        user_hashes = list()
        users_training = dict()
        users_testing = dict()

        all_users_dirs = os.listdir(collection_path)
        all_users_dirs = all_users_dirs[:num_of_users] if num_of_users >= 0 else all_users_dirs

        for user_dir in all_users_dirs:
            user_hash = user_dir
            user_hashes.append(user_hash)

            training_path = "{}/{}/{}_training.csv".format(collection_path, user_hash, user_hash)
            if verbosity >= 1:
                print("loading {}".format("{}/{}".format(collection_path, user_hash)))

            # load training set
            users_training[user_hash] = self.load_csv_file(training_path, verbosity)

            # load testing sets
            test_dirs = [test_dir for test_dir in os.listdir("{}/{}".format(collection_path, user_hash))
                         if not isfile("{}/{}/{}".format(collection_path, user_hash, test_dir))]
            test_dirs = test_dirs[:num_of_tests_per_user] if num_of_tests_per_user >= 0 else test_dirs

            users_testing[user_hash] = list()
            for test_dir in test_dirs:
                for test_file in os.listdir("{}/{}/{}".format(collection_path, user_hash, test_dir)):
                    users_testing[user_hash].append(self.load_csv_file("{}/{}/{}/{}".format(collection_path, user_hash, test_dir, test_file), verbosity))

        return user_hashes, users_training, users_testing
