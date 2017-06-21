from log.Print import *


def flatten_list(lst):
    flat_list = list();

    for el in lst:
        flat_list.extend(el)

    return flat_list

