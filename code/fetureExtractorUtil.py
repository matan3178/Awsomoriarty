from log.Print import *
from definitions import NUMBER_OF_REDUNDENT_LINES
import math


def start(data):
    normalize=remove_null_rows(remove_first_line(data))
    print(data_to_vector_size(normalize))
    print(derivation(normalize))


def remove_first_line(data):
    return data[NUMBER_OF_REDUNDENT_LINES:]


def remove_null_rows(list_of_lists):
    warning_flag = 0;
    result = list();
    for lst in list_of_lists:
        if 'null' in lst:
            warning_flag+=1
            if(warning_flag>1):
                print("More than one null in a row!",WARNING)
        else:
            result.append(lst)
            warning_flag=0
    return result


def data_to_vector_size(list_of_lists):
    result = list()
    for lst in list_of_lists:
        sum2 = math.sqrt(sum(x**2 for x in lst))
        result.append(sum2)
    return result


def derivation(list2):
    result=list()
    for i in range(len(list2)-1):
        temp=list()
        for j in range(len(list2[i])):
            temp.append(list2[i+1][j]-list2[i][j])
        result.append(temp)
    return result


print(derivation([[1,2,3],[-5,5,6],[7,8,9]]))
print(data_to_vector_size([[1,2,3],[-5,5,6],[7,8,9]]))
print(remove_null_rows([[1,2,3],[4,'null',6],[7,6,5],[1,'null','null'],[4,5,'null'],[1,2,3],[4,'null',6],[7,6,5],[1,'null','null']]));
print(remove_first_line([[1,2,3],[-5,5,6],[7,8,9]]))