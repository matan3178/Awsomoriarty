import os
from code.log.Print import *

print("initializing definitions...", COMMENT)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MORIARTY_DIR = ROOT_DIR + "/data/MoriartyDatasets/MoriartyDatasets1"
COLLECTION3V2_DIR = ROOT_DIR + "/../data/collection3V2"
NUMBER_OF_REDUNDENT_LINES = 1
NUMBER_OF_REDUNDENT_COLUMNS = 2

print("ROOT_DIR = " + ROOT_DIR, COMMENT)
print("MORIARTY_DIR = " + MORIARTY_DIR, COMMENT)
print("COLLECTION3V6_DIR = " + COLLECTION3V2_DIR, COMMENT)

# debug:
LIGHT_LOADING_num_of_users = 1              # -1 for all users
LIGHT_LOADING_num_of_tests_per_user = 2     # -1 for all test files
VERBOSITY_general = 2
VERBOSITY_training_autoencoder = 2
