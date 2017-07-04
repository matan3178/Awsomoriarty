from code.FetureExtractorUtil import remove_redundent_rows
from log.Print import *
import definitions
from code.FileLoader import FileLoader


class DataCenter:

    file_loader = FileLoader()

    user_hashes = "UNINITIALIZED"
    users_training = "UNINITIALIZED"
    users_testing = "UNINITIALIZED"

    HASH_INDEX_IN_SAMPLE = 0
    TIMESTAMP_INDEX_IN_SAMPLE = 1

    def __init__(self):
        return

    def load_data_collection3v2(self):

        self.user_hashes, all_users_training, all_users_all_testing_contiguous = self.file_loader.load_collection3v2(definitions.COLLECTION3V2_DIR, verbosity=2, num_of_users=1, num_of_tests_per_user=1)

        self.users_training = dict()
        for h in self.user_hashes:
            self.users_training[h] = remove_redundent_rows(all_users_training[h])

        self.users_testing = dict()
        for h in self.user_hashes:
            single_user_all_testing_contiguous = all_users_all_testing_contiguous[h]
            single_user_all_testing_split = list()
            real_user_hash = h

            for single_user_single_testing in single_user_all_testing_contiguous:
                single_user_single_testing = remove_redundent_rows(single_user_single_testing)

                user_samples = [sample for sample in single_user_single_testing
                                if sample[self.HASH_INDEX_IN_SAMPLE] == real_user_hash]

                thief_samples = [sample for sample in single_user_single_testing
                                 if sample[self.HASH_INDEX_IN_SAMPLE] != real_user_hash]

                single_user_all_testing_split.append(list([user_samples, thief_samples]))

            self.users_testing[h] = single_user_all_testing_split

        return self.user_hashes, self.users_training, self.users_testing
