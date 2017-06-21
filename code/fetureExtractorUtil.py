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


def derivation2(list2):
    result = list()
    for pair in [list2[i: i + 2] for i in range(len(list2)-1)]:
        temp = list()
        for x1,x2 in zip(pair[0], pair[1]):
            temp.append(x2-x1)
        result.append(temp)
    return result

print(derivation2([[1,2,3],[-5,5,6],[7,8,9]]))