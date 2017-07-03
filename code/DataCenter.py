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
        self.user_hashes, self.users_training, all_users_all_testing_contiguous = self.file_loader.load_collection3v2(definitions.COLLECTION3V2_DIR)

        for h in self.user_hashes:
            with all_users_all_testing_contiguous[h] as single_user_all_testing_contiguous:

                single_user_single_testing_split = [[], []]

                for single_user_single_testing in single_user_all_testing_contiguous:
                    real_user_hash = single_user_all_testing_contiguous[0][self.HASH_INDEX_IN_SAMPLE]

                    user_data = [sample for sample in single_user_single_testing
                                 if sample[self.HASH_INDEX_IN_SAMPLE] == real_user_hash]

                    thief_data = [sample for sample in single_user_single_testing
                                   if sample[self.HASH_INDEX_IN_SAMPLE] != real_user_hash]

                    single_user_single_testing_split[0].append(user_data)
                    single_user_single_testing_split[1].append(thief_data)



        return
