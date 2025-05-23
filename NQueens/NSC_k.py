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
    n = len(var) - 1
    map_register = [[0 for j in range(0, k + 1)] for i in range(0,n)]

    for i in range(1, n):
        for j in range(1, min(i, k) + 1):
            id_variable += 1
            map_register[i][j] = id_variable
    print(map_register)


    # (1): If a bit is true, the first bit of the corresponding register is true
    for i in range(1, n):
        plus_clause([-1 * var[i], map_register[i][1]])

    # (2): R[i - 1][j] = 1, R[i][j] = 1;
    for i in range(2, n):
        for j in range(1, min(i - 1, k) + 1):
            plus_clause([-1 * map_register[i - 1][j], map_register[i][j]])

    # (3): If bit i is on and R[i - 1][j - 1] = 1, R[i][j] = 1;
    for i in range(2, n):
        for j in range(2, min(i, k) + 1):
            plus_clause([-1 * var[i], -1 * map_register[i - 1][j - 1], map_register[i][j]])

    # (4): If bit i is off and R[i - 1][j] = 0, R[i][j] = 0;
    for i in range(2, n):
        for j in range(1, min(i - 1, k) + 1):
            plus_clause([var[i], map_register[i - 1][j], -1 * map_register[i][j]])

    # (5): If bit i is off, R[i][i] = 0;
    for i in range(1, k + 1):
        plus_clause([var[i], -1 * map_register[i][i]])

    # (6): If R[i - 1][j - 1] = 0, R[i][j] = 0;
    for i in range(2, n):
        for j in range(2, min(i, k) + 1):
            plus_clause([map_register[i - 1][j - 1], -1 * map_register[i][j]])

    # (7): (At least k) R[n - 1][k] = 1 or (n-th bit is true and R[n - 1][k - 1] = 1)
    plus_clause([map_register[n - 1][k], var[n]])
    plus_clause([map_register[n - 1][k], map_register[n - 1][k - 1]])
    # plus_clause([map_register[n - 1][k - 1]])

    # (8): (At most k) If i-th bit is true, R[i - 1][k] = 0;
    for i in range(k + 1, n + 1):
        plus_clause([-1 * var[i], -1 * map_register[i - 1][k]])

exactly_k([0,1,2,3,4], 2)