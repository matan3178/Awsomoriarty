from log.Print import *

def derivation(list2):
    result=list()
    for i in range(len(list2)-1):
        temp=list()
        for j in range(len(list2[i])):
            temp.append(list2[i+1][j]-list2[i][j])
        result.append(temp)
    return result

print(derivation([[1,2,3],[-5,5,6],[7,8,9]]))