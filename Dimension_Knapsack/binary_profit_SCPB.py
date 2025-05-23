from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.solvers import Glucose3
from typing import List

sat_solver = Glucose3()

def pos_i(i, k, weight):
    if i == 0:
        return 0
    if i < k:
        sum_w = sum(weight[1:i+1])
        return min(k, sum_w)
    else:
        return k

def plus_clause(clause):
    sat_solver.add_clause(clause)

def atMost_k(vars: List[int], weight: List[int], k):
    n = len(vars) - 1
    id_variable = n

    # Create map_register to hold the auxiliary variables
    map_register = [[0 for _ in range(k + 1)] for _ in range(n + 1)]

    for i in range(1, n):
        n_bits = pos_i(i, k, weight)
        for j in range(1, n_bits + 1):
            id_variable += 1
            map_register[i][j] = id_variable

    # print("Map register:")
    # print(map_register)

    # (1) X_i -> R_i,j for j = 1 to w_i
    for i in range(1, n):
        for j in range(1, weight[i] + 1):
            if j <= pos_i(i, k, weight):
                plus_clause([-vars[i], map_register[i][j]])

    # (2) R_{i-1,j} -> R_i,j for j = 1 to pos_{i-1}
    for i in range(2, n):
        for j in range(1, pos_i(i - 1, k, weight) + 1):
            plus_clause([-map_register[i - 1][j], map_register[i][j]])

    # (3) X_i ^ R_{i-1,j} -> R_i,j+w_i for j = 1 to pos_{i-1}
    for i in range(2, n):
        for j in range(1, pos_i(i - 1, k, weight) + 1):
            if j + weight[i] <= k and j + weight[i] <= pos_i(i, k, weight):
                plus_clause([-vars[i], -map_register[i - 1][j], map_register[i][j + weight[i]]])

    # (8) At Most K: X_i -> ¬R_{i-1,k+1-w_i} for i = 2 to n 
    for i in range(2, n + 1):
        if k + 1 - weight[i] > 0 and k + 1 - weight[i] <= pos_i(i - 1, k, weight):
            plus_clause([-vars[i], -map_register[i - 1][k + 1 - weight[i]]])

    return id_variable

def atLeast_k(vars: List[int], weight: List[int], k, max_var):
    n = len(vars) - 1
    id_variable = max_var

    # Create map_register to hold the auxiliary variables
    map_register = [[0 for _ in range(k + 1)] for _ in range(n + 1)]

    for i in range(1, n):
        n_bits = pos_i(i, k, weight)
        for j in range(1, n_bits + 1):
            id_variable += 1
            map_register[i][j] = id_variable

    # print("Map register:")
    # print(map_register)

    # (1) X_i -> R_i,j for j = 1 to w_i
    for i in range(1, n):
        for j in range(1, weight[i] + 1):
            if j <= pos_i(i, k, weight):
                plus_clause([-vars[i], map_register[i][j]])

    # (2) R_{i-1,j} -> R_i,j for j = 1 to pos_{i-1}
    for i in range(2, n):
        for j in range(1, pos_i(i - 1, k, weight) + 1):
            plus_clause([-map_register[i - 1][j], map_register[i][j]])

    # (3) X_i ^ R_{i-1,j} -> R_i,j+w_i for j = 1 to pos_{i-1}
    for i in range(2, n):
        for j in range(1, pos_i(i - 1, k, weight) + 1):
            if j + weight[i] <= k and j + weight[i] <= pos_i(i, k, weight):
                plus_clause([-vars[i], -map_register[i - 1][j], map_register[i][j + weight[i]]])

    # (4) ¬X_i ^ ¬R_{i-1,j} -> ¬R_i,j for j = 1 to pos_{i-1}
    for i in range(2, n):
        for j in range(1, pos_i(i - 1, k, weight) + 1):
            plus_clause([vars[i], map_register[i - 1][j], -map_register[i][j]])

    # (5) ¬X_i -> ¬R_i,j for j = 1 + pos_{i-1} to pos_i
    for i in range(1, n):
        # if pos_i(i - 1, k, weight) < k:
            for j in range(1 + pos_i(i - 1, k, weight), pos_i(i, k, weight) + 1):
                plus_clause([vars[i], -map_register[i][j]])

    # (6) ¬R_{i-1,j} -> ¬R_i,j+w_i for j = 1 to pos_{i-1}
    for i in range(2, n):
        # if pos_i(i - 1, k, weight) < k:
            for j in range(1, pos_i(i - 1, k, weight) + 1):
                if j + weight[i] <= k and j + weight[i] <= pos_i(i, k, weight):
                    plus_clause([map_register[i - 1][j], -map_register[i][j + weight[i]]])

    # (7) R_{n-1,k} v (X_n ^ R_{n-1,k-w_n})
    if k > pos_i(n - 1, k, weight):
        plus_clause([vars[n]])
        plus_clause([map_register[n - 1][k - weight[n]]])
    else:
        plus_clause([map_register[n - 1][k], vars[n]])
        if k - weight[n] > 0 and k - weight[n] <= pos_i(n - 1, k, weight):
            plus_clause([map_register[n - 1][k], map_register[n - 1][k - weight[n]]])
    return id_variable

def positive_range(end):
    if (end < 0):
        return []
    return range(end)

def sum_profit(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
    return sum

def binary_profit(min, max, weights, max_weight, profits):
    while max-min > 1:
        mid = int(min + (max - min) / 2)
        sat_solver = Glucose3()
        status = pysat_solution(weights, max_weight, profits, mid)
        sat_solver.delete()
        print(status)
        if (status == "sat"):
            min = mid
            sat_solver.delete()
        elif (status == "timeout"):
            sat_solver.delete()
            continue
        else:
            max = mid
            sat_solver.delete()

        print("Mid:", mid)

def pysat_solution(weights, max_weight, profits, min_profit):

    n = len(weights)-1
    weight_vars = list(range(n + 1))
    profit_vars = list(range(n + 1))

    max_var  = atMost_k(weight_vars, weights, max_weight)

    max_var = atLeast_k(profit_vars, profits, min_profit,max_var)

    
    num_vars = sat_solver.nof_vars()
    num_clauses = sat_solver.nof_clauses()

    print("Variables: " + str(num_vars))
    print("Clauses:" + str(num_clauses))

    sat_status = sat_solver.solve_limited(expect_interrupt = True)

    if sat_status is False:
        print("No solutions found")
        return "unsat"
    else:
        solution = sat_solver.get_model()
        if solution is None:
            print("time out")
            return "timeout"
        else:
            solution = sat_solver.get_model()
            print(f"Solution found: {solution}")
            for i, val in enumerate(solution, start=1):
                if i <= n:
                    print(f"X{i} = {int(val > 0)}")
            return "sat"

        
   
    

def opp_solution(rectangles, strip):
    cnf = CNF()
    variables = {}
    counter = 1
    for i in range(len(rectangles)):
        for j in range(len(rectangles)):
            variables[f"lr{i + 1},{j + 1}"] = counter  # lri,rj
            counter += 1
            variables[f"ud{i + 1},{j + 1}"] = counter  # uri,rj
            counter += 1
        for e in positive_range(strip[0] - rectangles[i][0] + 2):
            variables[f"px{i + 1},{e}"] = counter  # pxi,e
            counter += 1
        for f in positive_range(strip[1] - rectangles[i][1] + 2):
            variables[f"py{i + 1},{f}"] = counter  # pyi,f
            counter += 1

    # Add the 2-literal axiom clauses
    for i in range(len(rectangles)):
        for e in range(strip[0] - rectangles[i][0] + 1):  # -1 because we're using e+1 in the clause
            cnf.append([-variables[f"px{i + 1},{e}"], variables[f"px{i + 1},{e + 1}"]])
        for f in range(strip[1] - rectangles[i][1] + 1):  # -1 because we're using f+1 in the clause
            cnf.append([-variables[f"py{i + 1},{f}"], variables[f"py{i + 1},{f + 1}"]])

    # Add the 4-literal axiom clauses
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            cnf.append([variables[f"lr{i + 1},{j + 1}"], variables[f"lr{j + 1},{i + 1}"], variables[f"ud{i + 1},{j + 1}"],
                        variables[f"ud{j + 1},{i + 1}"]])

    # Add the 3-literal non-overlapping constraints
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            for e in positive_range(strip[0] - rectangles[i][0]) :
                if f"px{j + 1},{rectangles[i][0] - 1}" in variables:
                    cnf.append([-variables[f"lr{i + 1},{j + 1}"], -variables.get(f"px{j + 1},{rectangles[i][0] - 1}", 0)])
                if(strip[0] - rectangles[i][0] - rectangles[j][0] > 0) and f"px{i + 1},{strip[0] - rectangles[i][0] - rectangles[j][0]}" in variables:
                    cnf.append([-variables[f"lr{i + 1},{j + 1}"],
                                variables[f"px{i + 1},{strip[0] - rectangles[i][0] - rectangles[j][0]}"]])
                if f"px{j + 1},{e + rectangles[i][0]}" in variables:
                    cnf.append([-variables[f"lr{i + 1},{j + 1}"], variables[f"px{i + 1},{e}"],
                                -variables.get(f"px{j + 1},{e + rectangles[i][0]}", 0)])
                if f"px{i + 1},{e + rectangles[j][0]}" in variables:
                    cnf.append([-variables[f"lr{j + 1},{i + 1}"], variables.get(f"px{j + 1},{e}", 0),
                                -variables.get(f"px{i + 1},{e + rectangles[j][0]}", 0)])
                    
            for f in positive_range(strip[1] - rectangles[i][1]):
                if f"py{j + 1},{rectangles[i][1] - 1}" in variables:
                    cnf.append([-variables[f"ud{i + 1},{j + 1}"], -variables.get(f"py{j + 1},{rectangles[i][1] - 1}", 0)])
                if strip[1] - rectangles[i][1] - rectangles[j][1] > 0:
                    cnf.append([-variables[f"ud{i + 1},{j + 1}"],
                                variables[f"py{i + 1},{strip[1] - rectangles[i][1] - rectangles[j][1]}"]])
                if f"py{j + 1},{f + rectangles[i][1]}" in variables:
                    cnf.append([-variables[f"ud{i + 1},{j + 1}"], variables[f"py{i + 1},{f}"],
                            -variables.get(f"py{j + 1},{f + rectangles[i][1]}", 0)])

                if f"py{i + 1},{f + rectangles[j][1]}" in variables:
                    cnf.append([-variables[f"ud{j + 1},{i + 1}"], variables.get(f"py{j + 1},{f}", 0),
                                -variables.get(f"py{i + 1},{f + rectangles[j][1]}", 0)])

    for i in range(len(rectangles)):
        cnf.append([variables[f"px{i + 1},{strip[0] - rectangles[i][0]}"]])  # px(i, W-wi)
        if f"py{i + 1},{strip[1] - rectangles[i][1]}" in variables:
            cnf.append([variables[f"py{i + 1},{strip[1] - rectangles[i][1]}"]])  # py(i, H-hi)

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            # if indomain(len(px[j]) - 1, width_rects[i] - 1)
            if strip[0] - rectangles[i][0] + 1 >= rectangles[j][0] - 1:
                cnf.append([-variables[f"lr{i + 1},{j + 1}"], -variables[f"px{j + 1},{rectangles[i][0] - 1}"]])
            else:
                if f"px{j + 1},{strip[0] - rectangles[i][0] + 1}" in variables:
                    cnf.append([-variables[f"lr{i + 1},{j + 1}"], variables[f"px{j + 1},{strip[0] - rectangles[i][0] + 1}"]])
            # if indomain(len(px[i] - 1, width_rects[j] - 1)
            if strip[0] - rectangles[j][0] + 1 >= rectangles[i][0] - 1:
                cnf.append([-variables[f"lr{j + 1},{i + 1}"], -variables[f"px{i + 1},{rectangles[j][0] - 1}"]])
            else:
                cnf.append([-variables[f"lr{j + 1},{i + 1}"], variables[f"px{i + 1},{strip[0] - rectangles[j][0] + 1}"]])
            # if indomain(len(py[j]) - 1, height_rects[i] - 1)
            if strip[1] - rectangles[i][1] + 1 >= rectangles[j][1] - 1:
                cnf.append([-variables[f"ud{i + 1},{j + 1}"], -variables[f"py{j + 1},{rectangles[i][1] - 1}"]])
            else:
                if f"py{j + 1},{strip[1] - rectangles[i][1] + 1}" in variables:
                    cnf.append([-variables[f"ud{i + 1},{j + 1}"], variables[f"py{j + 1},{strip[1] - rectangles[i][1] + 1}"]])
            # if indomain(len(py[i]) - 1, height_rects[j] - 1)
            if strip[1] - rectangles[j][1] + 1 >= rectangles[i][1] - 1:
                cnf.append([-variables[f"ud{j + 1},{i + 1}"], -variables[f"py{i + 1},{rectangles[j][1] - 1}"]])
            else:
                if f"py{i + 1},{strip[1] - rectangles[j][1] + 1}" in variables:
                    cnf.append([-variables[f"ud{j + 1},{i + 1}"], variables[f"py{i + 1},{strip[1] - rectangles[j][1] + 1}"]])


    with Solver(name="mc") as solver:
        solver.append_formula(cnf)
        if solver.solve():
            pos = [[0 for i in range(2)] for j in range(len(rectangles))]
            model = solver.get_model()
            result = {}
            for var in model:
                if var > 0:
                    result[list(variables.keys())[list(variables.values()).index(var)]] = True
                else:
                    result[list(variables.keys())[list(variables.values()).index(-var)]] = False

            for i in range(len(rectangles)):
                for e in range(strip[0] - rectangles[i][0] + 1):
                    if result[f"px{i + 1},{e}"] == False and result[f"px{i + 1},{e + 1}"] == True:
                        # print(f"x{i + 1} = {e + 1}")
                        pos[i][0] = e + 1
                    if e == 0 and result[f"px{i + 1},{e}"] == True:
                        # print(f"x{i + 1} = 0")
                        pos[i][0] = 0
                for f in range(strip[1] - rectangles[i][1] + 1):
                    if result[f"py{i + 1},{f}"] == False and result[f"py{i + 1},{f + 1}"] == True:
                        # print(f"y{i + 1} = {f + 1}")
                        pos[i][1] = f + 1
                    if f == 0 and result[f"py{i + 1},{f}"] == True:
                        # print(f"y{i + 1} = 0")
                        pos[i][1] = 0
                        
            print(["sat", pos])
            return "sat"
        else:
            print("unsat")
            return "unsat"
    

def solve_problem():

    weights = [0, 10, 20, 30, 40, 50]
    max_weight = 30
    profits = [0, 30, 30, 30, 30, 30]

    total = sum_profit(profits)

    pysat_solution(weights, max_weight, profits, 60)

    # binary_profit(1, total, weights, max_weight, profits)

solve_problem()