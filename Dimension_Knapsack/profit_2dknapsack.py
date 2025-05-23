from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.solvers import Glucose3
from pypblib import pblib
from pypblib.pblib import PBConfig, Pb2cnf
# OOP using binary search

def positive_range(end):
    if (end < 0):
        return []
    return range(end)

def sum_profit(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
    return sum

def binary_profit(min, max):
    while max-min > 1:
        mid = int(min + (max - min) / 2)
        status = pysat_solution(mid)
        if (status == "sat"):
            min = mid
        else:
            max = mid
        print("Mid:", mid)

def pysat_solution(max_profit):
    # <Loop> First line is bound weight and bound profit, two next lines is weight and height of each item
    with open('base_input.txt', 'r') as file:
        lines = file.readlines()
        len_file = len(lines)
        for i in range(0, len_file,5):
            bounds = list(map(int,lines[i].strip().split()))
            weights = list(map(int, lines[i + 1].strip().split()))
            heights = list(map(int, lines[i + 2].strip().split()))
            widths = list(map(int, lines[i + 3].strip().split()))
            profits = list(map(int, lines[i + 4].strip().split()))
            strip_width = bounds[2]
            strip_height = bounds[3]
            strip = (strip_width, strip_height)
            weight_bound = bounds[0]
            num_items = len(weights)
            formula = []
            rectangles = []
            vars = list(range(1, num_items + 1))

            solver = Glucose3()
            pbConfig = PBConfig()
            pbConfig.set_PB_Encoder(pblib.PB_BEST)
            pb2 = Pb2cnf(pbConfig)

            max_var = pb2.encode_leq(weights, vars, weight_bound, formula, num_items+1)
            # print(max_var)
            max_var = pb2.encode_geq(profits, vars, max_profit, formula, max_var+1)

            for clause in formula:
                solver.add_clause(clause)

            num_vars = solver.nof_vars()
            num_clauses = solver.nof_clauses()

            print("Variables: " + str(num_vars))
            print("Clauses:" + str(num_clauses))

            sat_status = solver.solve_limited(expect_interrupt = True)

            if sat_status is False:
                print("No solutions found")
                return "unsat"
            else:
                solution = solver.get_model()
                if solution is None:
                    print("time out")
                    return "timeout"
                else:
                    model = solver.get_model()
                    # print(model)
                    for i in range(1, num_items+1):
                        temp = ()
                        if model[i - 1] > 0:
                            temp = (widths[i-1],heights[i-1])
                        if temp != (): rectangles.append(temp)
                    # print(rectangles)
                    selected_items = [i for i in range(1, num_items + 1) if model[i-1] > 0]
                    # print("\nSelected Items:")
                    # print(selected_items)
                    opp_solution(rectangles)
                    rectangles = []
                    return "sat"

def opp_solution(rectangles):
    strip = (100, 100)
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
    profits = []
    with open('base_input.txt', 'r') as file:
        lines = file.readlines()
        len_file = len(lines)
        for i in range(0, len_file,5):
            profits = list(map(int, lines[i + 4].strip().split()))
    total = sum_profit(profits)
    binary_profit(1, total)
solve_problem()
# opp_solution()