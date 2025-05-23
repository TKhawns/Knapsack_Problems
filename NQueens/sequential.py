from pysat.formula import CNF
import numpy as np

def AMO_binary_encoding(n):
    cnf = CNF()
    # number of variables need to add
    s = n - 1
    print("S: ",s)

    original_variable = [i + 1 for i in range(n)]
    add_variable = [n+ i + 1 for i in range(s)]

    # Add clauses
    for i in range(s):
        cnf.append([-original_variable[i], add_variable[i]])
    for i in range(1,n):
        cnf.append([-original_variable[i], -add_variable[i-1]])
    for i in range(1,s):
        cnf.append([-add_variable[i-1], add_variable[i]])

    return cnf

def ALO_binary_encoding(n):
    cnf = CNF()
    original_variable = [i + 1 for i in range(n)]
    cnf.append(original_variable)
    return cnf

n = 5
cnf_AMO = AMO_binary_encoding(n)
cnf_ALO = ALO_binary_encoding(n)
print(len(cnf_AMO.clauses)) # must be 3n-1
for clause in cnf_AMO.clauses:
    print(clause)