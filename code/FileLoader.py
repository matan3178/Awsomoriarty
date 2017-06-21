from log.Print import *
import csv


class FileLoader:

    use_cache = True
    cache = dict()

    def __init__(self, use_cache):
        self.use_cache = use_cache
        return

    def __init__(self):
        return

    def load_csv_file(self, path):
        if self.use_cache and path in self.cache:
            return self.cache[path]
        else:
            data = list()
            with open(path) as datafile:
                lines = enumerate(csv.reader(datafile, delimiter=","))
                data.extend(lines)
            if self.use_cache: self.cache[path] = data
            return data