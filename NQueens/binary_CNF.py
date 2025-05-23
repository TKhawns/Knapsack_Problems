from pysat.formula import CNF
import numpy as np

def AMO_binary_encoding(n):
    cnf = CNF()
    # number of variables need to add
    k = (n - 1).bit_length()
    print("K: ",k)

    temp = np.unpackbits(np.arange(8).astype(np.uint8)[:, None], axis = 1)[:, -k:]
    print("temp:\n", temp)

    original_variable = [i + 1 for i in range(n)]
    add_variable = [n+ i + 1 for i in range(k)]

    for i in range(n):
        for j in range(k-1,-1,-1):
            if (temp[i][j] == 0):
                cnf.append([-original_variable[i], -add_variable[k-j-1]])
            else:
                cnf.append([-original_variable[i], add_variable[k-j-1]])
    return cnf

def ALO_binary_encoding(n):
    cnf = CNF()
    original_variable = [i + 1 for i in range(n)]
    cnf.append(original_variable)
    return cnf

n = 5
cnf_AMO = AMO_binary_encoding(n)
cnf_ALO = ALO_binary_encoding(n)

for clause in cnf_AMO.clauses:
    print(clause)