from log.Print import *
import math

def removeNullRows(list_of_lists):
    warningFlag=0;
    result=list();
    for lst in list_of_lists:
        if 'null' in lst:
            warningFlag+=1
            if(warningFlag>1):
                print("More than one null in a row!",WARNING)
        else:
            result.append(lst)
            warningFlag=0
    return result

def dataToVectorSize(list_of_lists):
    result=list()
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
print(dataToVectorSize([[1,2,3],[-5,5,6],[7,8,9]]))
print(removeNullRows([[1,2,3],[4,'null',6],[7,6,5],[1,'null','null'],[4,5,'null'],[1,2,3],[4,'null',6],[7,6,5],[1,'null','null']]));