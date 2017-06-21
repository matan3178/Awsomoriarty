import definitions
from log.Print import *
from code.FileLoader import FileLoader


def do_something():
    fl = FileLoader()
    fl.load_csv_file(definitions.MORIARTY_DIR + str("0a50e09262.csv"))
    print(fl)
    return


def main():
    print("Welcome to Awesomoriarty!", HEADER)
    do_something()
    print("bye bye", HEADER)
    return

if __name__ == "__main__":
    main()