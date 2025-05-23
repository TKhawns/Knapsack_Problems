from typing import List
from pysat.formula import CNF
from pysat.solvers import Glucose3, Solver

id_variable = 0
sat_solver = Glucose3()

def plus_clause(clause):
    sat_solver.add_clause(clause)
    print(clause)

def exactly_k(var: List[int], k):
    global id_variable
    cnf = CNF()
    n = len(var)

    # Init register R(1) to R(n-1)
    map_register = []
    for i in range(0,n-1):
        temp = []
        for j in range(k+1):
            temp.append(0)
        map_register.append(temp)

    for i in range(1, n-1):
        for j in range(1, min(i, k) + 1):
            id_variable += 1
            map_register[i][j] = id_variable

    # Formula (1)
    for i in range(0, n-1):
        cnf.append([-var[i], map_register[i][1]])

    # Formula (2)
    for i in range(1, n-1):
        for j in range(0, min(i-1,k) + 1):
            cnf.append([-map_register[i-1][j], map_register[i][j]])

    # Formula (3)
    for i in range(1,n-1):
        for j in range(1, min(i,k)+1):
            cnf.append([-var[i], -map_register[i-1][j-1], map_register[i][j]])

    # Formula (4)
    for i in range(1, n-1):
        for j in range(0, min(i-1,k) + 1):
            cnf.append([var[i], map_register[i-1][j], -map_register[j][j]])

    # Formula (5)
    for i in range(0, k):
        cnf.append([var[i], -map_register[i][j]])
    
    # Formula (6)
    for i in range(1, n-1):
        for j in range(1, min(i,k) + 1):
            cnf.append([map_register[i-1][j-1], -map_register[i][j]])
    
    # Formula (7)
    cnf.append([map_register[n-2][k-1],var[n-1]])
    cnf.append([map_register[n-2][k-1],map_register[n-2][k-2]])

    # Formula (8)
    for i in range(k, n):
        cnf.append([-var[i], -map_register[i-1][k-1]])

    print(cnf.clauses)

def at_most_k(var: List[int], k):
    cnf = CNF()
    n = len(var)
    global id_variable

    # Init register R(1) to R(n-1)
    map_register = []
    for i in range(0,n-1):
        temp = []
        for j in range(k+1):
            temp.append(0)
        map_register.append(temp)

    for i in range(1, n-1):
        for j in range(1, min(i, k) + 1):
            id_variable += 1
            map_register[i][j] = id_variable
    # Formula (1)
    for i in range(0, n-1):
        cnf.append([-var[i], map_register[i][1]])

    # Formula (2)
    for i in range(1, n-1):
        for j in range(0, min(i-1,k) + 1):
            cnf.append([-map_register[i-1][j], map_register[i][j]])

    # Formula (3)
    for i in range(1,n-1):
        for j in range(1, min(i,k)+1):
            cnf.append([-var[i], -map_register[i-1][j-1], map_register[i][j]])

    # Formula (8)
    for i in range(k, n):
        cnf.append([-var[i], -map_register[i-1][k-1]])
    print(cnf.clauses)


def at_least_k(var: List[int], k):
    global id_variable
    cnf = CNF()
    n = len(var)

    # Init register R(1) to R(n-1)
    map_register = []
    for i in range(0,n-1):
        temp = []
        for j in range(k+1):
            temp.append(0)
        map_register.append(temp)

    for i in range(1, n-1):
        for j in range(1, min(i, k) + 1):
            id_variable += 1
            map_register[i][j] = id_variable

    # Formula (1)
    for i in range(0, n-1):
        cnf.append([-var[i], map_register[i][1]])

    # Formula (2)
    for i in range(1, n-1):
        for j in range(0, min(i-1,k) + 1):
            cnf.append([-map_register[i-1][j], map_register[i][j]])

    # Formula (3)
    for i in range(1,n-1):
        for j in range(1, min(i,k)+1):
            cnf.append([-var[i], -map_register[i-1][j-1], map_register[i][j]])

    # Formula (4)
    for i in range(1, n-1):
        for j in range(0, min(i-1,k) + 1):
            cnf.append([var[i], map_register[i-1][j], -map_register[j][j]])

    # Formula (5)
    for i in range(0, k):
        cnf.append([var[i], -map_register[i][j]])
    
    # Formula (6)
    for i in range(1, n-1):
        for j in range(1, min(i,k) + 1):
            cnf.append([map_register[i-1][j-1], -map_register[i][j]])
    
    # Formula (7)
    cnf.append([map_register[n-2][k-1],var[n-1]])
    cnf.append([map_register[n-2][k-1],map_register[n-2][k-2]])
    
    print(cnf.clauses)

def range_k(var: List[int], u, v):
    global id_variable
    cnf = CNF()
    n = len(var)

    # Init register R(1) to R(n-1)
    map_register = []
    for i in range(0,n-1):
        temp = []
        for j in range(v+1):
            temp.append(0)
        map_register.append(temp)

    for i in range(1, n-1):
        for j in range(1, min(i,v) + 1):
            id_variable += 1
            map_register[i][j] = id_variable

    # Formula (1)
    for i in range(0, n-1):
        cnf.append([-var[i], map_register[i][1]])

    # Formula (2)
    for i in range(1, n-1):
        for j in range(0, min(i-1,v) + 1):
            cnf.append([-map_register[i-1][j], map_register[i][j]])

    # Formula (3)
    for i in range(1,n-1):
        for j in range(1, min(i,v)+1):
            cnf.append([-var[i], -map_register[i-1][j-1], map_register[i][j]])

    # Formula (4)
    for i in range(1, n-1):
        for j in range(0, min(i-1,u) + 1):
            cnf.append([var[i], map_register[i-1][j], -map_register[j][j]])

    # Formula (5)
    for i in range(0, u):
        cnf.append([var[i], -map_register[i][j]])
    
    # Formula (6)
    for i in range(1, n-1):
        for j in range(1, min(i,u) + 1):
            cnf.append([map_register[i-1][j-1], -map_register[i][j]])
    
    # Formula (7)
    cnf.append([map_register[n-2][v-1],var[n-1]])
    cnf.append([map_register[n-2][v-1],map_register[n-2][v-2]])

    # Formula (8)
    for i in range(v, n):
        cnf.append([-var[i], -map_register[i-1][v-1]])

    print(cnf.clauses)



# at_most_k([1,2,3],2)
# exactly_k([1,2,3], 2)
# at_least_k([1,2,3],2)
# range_k([1,2,3],1,2)

exactly_k([0,1,2,3,4], 2)
exactly_k([1,2,3,4], 2)