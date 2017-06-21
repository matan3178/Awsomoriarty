import definitions
from log.Print import *
from code.FileLoader import FileLoader


def do_something():
    fl = FileLoader(use_cache=False)
    fl.load_collection2v3(definitions.COLLECTION3V6_DIR)
    return


def main():
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)
    return

if __name__ == "__main__":
    main()
