import os
from log.Print import *
print("initializing definitions...", COMMENT)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MORIARTY_DIR = ROOT_DIR + "/data/MoriartyDatasets/MoriartyDatasets1/"
COLLECTION3V6_DIR = ROOT_DIR + "/data/collection3V2/"

print("ROOT_DIR = " + ROOT_DIR, COMMENT)
print("MORIARTY_DIR = " + MORIARTY_DIR, COMMENT)
print("COLLECTION3V6_DIR = " + COLLECTION3V6_DIR, COMMENT)
