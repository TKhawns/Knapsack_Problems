from pysat.formula import CNF
import numpy as np
import math

def AMO_binomial(arr):
    n = len(arr)
    cnf = CNF()
    cnf.append(arr)
    for i in range(n):
        for j in range(i+1,n):
            cnf.append([-arr[i], -arr[j]])
    return cnf



def ALO_encoding(n):
    cnf = CNF()
    original_variable = [i + 1 for i in range(n)]
    cnf.append(original_variable)
    return cnf


def AMO_commander_encoding(n):
    cnf = CNF()
    # number of variables need to add
    c = int(math.sqrt(n))
    print("C: ",c)

    all_group = []
    original_variable = [i + 1 for i in range(n)]
    group_variable = [n+ i + 1 for i in range(c)]
    
    for i in range(0, n, c):
        subarray = original_variable[i:i + c]
        all_group.append(subarray)

    if len(all_group) > 1 and len(all_group[-1]) < c:
        all_group[-2].extend(all_group[-1])
        all_group.pop()
    print(all_group)

    cnf = AMO_binomial(group_variable)
    for i in range(c):
        cnf.extend(AMO_binomial(all_group[i]))
    
    for i in range(c):
        cnf.append([-group_variable[i]])
    
    for i in range(c):
        for j in range(len(all_group[i])):
            cnf.append([group_variable[i], -all_group[i][j]])

    print(cnf.clauses)
    return cnf

# n = 5
# cnf_AMO = AMO_commander_encoding(n)
# cnf_ALO = ALO_encoding(n)
# print(len(cnf_AMO.clauses)) # must be 3n-1
# for clause in cnf_AMO.clauses:
#     print(clause)

AMO_commander_encoding(5)