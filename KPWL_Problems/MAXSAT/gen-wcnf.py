from pysat.formula import WCNF
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl import load_workbook
from zipfile import BadZipFile
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime
from pysat.examples.rc2 import RC2
import matplotlib.pyplot as plt
import time
import signal

id_counter = 0


def positive_range(end):
    if (end < 0):
        return []
    return range(end)

def pos_i(i, k, profit):
    if i == 0:
        return 0
    if i < k:
        sum_w = sum(profit[1:i+1])
        return min(k, sum_w)
    else:
        return k

def maxsat_constraints(rectangles, strip, profits, file_name, weights):
    # Define the variables
    cnf = WCNF()

    width = strip[0]
    height = strip[1]
    C = strip[2]
    variables = {}
    counter = 1
    n_scpb = len(rectangles)
    n = n_scpb

    weight2 = [0] + weights

    # a_i = 1 if item i is selected.
    for i in range(n):
        variables[f"a{i + 1}"] = counter
        counter += 1

    map_register2 = [[0 for _ in range(C + 1)] for _ in range(n_scpb + 1)]
    for i in range(1, n_scpb):
        n_bits = pos_i(i, C, weight2)
        for j in range(1, n_bits + 1):
            map_register2[i][j] = counter
            counter += 1
    # Weight constraints
    # (0) if weight[i] > k => x[i] False
    for i in range(1, n_scpb):
        if weight2[i] > C:
            cnf.append([-variables[f"a{i}"]])

    # (1) X_i -> R_i,j for j = 1 to w_i k
    for i in range(1, n_scpb):
        for j in range(1, weight2[i] + 1):
            if j <= pos_i(i, C, weight2):
                cnf.append([-variables[f"a{i}"], map_register2[i][j]])

    # (2) R_{i-1,j} -> R_i,j for j = 1 to pos_{i-1}
    for i in range(2, n_scpb):
        for j in range(1, pos_i(i - 1, C, weight2) + 1):
            cnf.append([-map_register2[i - 1][j], map_register2[i][j]])

    # (3) X_i ^ R_{i-1,j} -> R_i,j+w_i for j = 1 to pos_{i-1}
    for i in range(2, n_scpb):
        for j in range(1, pos_i(i - 1, C, weight2) + 1):
            if j + weight2[i] <= C and j + weight2[i] <= pos_i(i, C, weight2):
                cnf.append([-variables[f"a{i}"], -map_register2[i - 1][j], map_register2[i][j + weight2[i]]])

    # (8) At Most K: X_i -> ¬R_{i-1,k+1-w_i} for i = 2 to n 
    for i in range(2, n_scpb + 1):
        if C + 1 - weight2[i] > 0 and C + 1 - weight2[i] <= pos_i(i - 1, C, weight2):
            cnf.append([-variables[f"a{i}"], -map_register2[i - 1][C + 1 - weight2[i]]])

    # create lr, ud, px, py variables from 1 to N.
    # SAT Encoding of 2OPP
    for i in range(n):
        for j in range(n):
            if i != j:
                variables[f"lr{i + 1},{j + 1}"] = counter  # lri,rj
                counter += 1
                variables[f"ud{i + 1},{j + 1}"] = counter  # uri,rj
                counter += 1
        for e in range(width):
            variables[f"px{i + 1},{e}"] = counter  # pxi,e
            counter += 1
        for f in range(height):
            variables[f"py{i + 1},{f}"] = counter  # pyi,f
            counter += 1

    # Rotated variables
    for i in range(n):
        variables[f"r{i + 1}"] = counter
        counter += 1

    # Add the 2-literal axiom clauses (order constraint)
    # Formula (3).
    # [Update] if a_i = 1 -> order-encoding.
    for i in range(n):
        for e in range(width - 1):  # -1 because we're using e+1 in the clause
            cnf.append([-variables[f"a{i + 1}"], -variables[f"px{i + 1},{e}"],
                        variables[f"px{i + 1},{e + 1}"]])
        for f in range(height - 1):
            cnf.append([-variables[f"a{i + 1}"], -variables[f"py{i + 1},{f}"],
                        variables[f"py{i + 1},{f + 1}"]])
            
    # Add the 3-literal non-overlapping constraints
    # Formula (4).
    def non_overlapping(rotated, i, j, h1, h2, v1, v2):
        if not rotated:
            i_width = rectangles[i][0]
            i_height = rectangles[i][1]
            j_width = rectangles[j][0]
            j_height = rectangles[j][1]
            i_rotation = variables[f"r{i + 1}"]
            j_rotation = variables[f"r{j + 1}"]
        else:
            i_width = rectangles[i][1]
            i_height = rectangles[i][0]
            j_width = rectangles[j][1]
            j_height = rectangles[j][0]
            i_rotation = -variables[f"r{i + 1}"]
            j_rotation = -variables[f"r{j + 1}"]

        # Square symmertry breaking, if i is square than it cannot be rotated
        if i_width == i_height and rotated:
            i_square = True
            cnf.append([-variables[f"r{i + 1}"]])
        else:
            i_square = False

        if j_width == j_height and rotated:
            j_square = True
            cnf.append([-variables[f"r{j + 1}"]])
        else:
            j_square = False

        extra_cnf = [-variables[f"a{i + 1}"], -variables[f"a{j + 1}"]]
        # lri,j v lrj,i v udi,j v udj,i
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
        if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
        if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
        if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])

        cnf.append(extra_cnf + four_literal + [i_rotation])
        cnf.append(extra_cnf + four_literal + [j_rotation])

        # ¬lri, j ∨ ¬pxj, e
        if h1 and not i_square:
            for e in range(min(width, i_width)):
                    cnf.append(extra_cnf + [i_rotation,
                                -variables[f"lr{i + 1},{j + 1}"],
                                -variables[f"px{j + 1},{e}"]])
        # ¬lrj,i ∨ ¬pxi,e
        if h2 and not j_square:
            for e in range(min(width, j_width)):
                    cnf.append(extra_cnf + [j_rotation,
                                -variables[f"lr{j + 1},{i + 1}"],
                                -variables[f"px{i + 1},{e}"]])
        # ¬udi,j ∨ ¬pyj,f
        if v1 and not i_square:
            for f in range(min(height, i_height)):
                    cnf.append(extra_cnf + [i_rotation,
                                -variables[f"ud{i + 1},{j + 1}"],
                                -variables[f"py{j + 1},{f}"]])
        # ¬udj, i ∨ ¬pyi, f,
        if v2 and not j_square:
            for f in range(min(height, j_height)):
                    cnf.append(extra_cnf + [j_rotation,
                                -variables[f"ud{j + 1},{i + 1}"],
                                -variables[f"py{i + 1},{f}"]])

        for e in positive_range(width - i_width):
            # ¬lri,j ∨ ¬pxj,e+wi ∨ pxi,e
            if h1 and not i_square:
                    cnf.append(extra_cnf + [i_rotation,
                                -variables[f"lr{i + 1},{j + 1}"],
                                variables[f"px{i + 1},{e}"],
                                -variables[f"px{j + 1},{e + i_width}"]])

        for e in positive_range(width - j_width):
            # ¬lrj,i ∨ ¬pxi,e+wj ∨ pxj,e
            if h2 and not j_square:
                    cnf.append(extra_cnf + [j_rotation,
                                -variables[f"lr{j + 1},{i + 1}"],
                                variables[f"px{j + 1},{e}"],
                                -variables[f"px{i + 1},{e + j_width}"]])

        for f in positive_range(height - i_height):
            # udi,j ∨ ¬pyj,f+hi ∨ pxi,e
            if v1 and not i_square:
                    cnf.append(extra_cnf + [i_rotation,
                                -variables[f"ud{i + 1},{j + 1}"],
                                variables[f"py{i + 1},{f}"],
                                -variables[f"py{j + 1},{f + i_height}"]])
        for f in positive_range(height - j_height):
            # ¬udj,i ∨ ¬pyi,f+hj ∨ pxj,f
            if v2 and not j_square:
                cnf.append(extra_cnf + [j_rotation,
                            -variables[f"ud{j + 1},{i + 1}"],
                            variables[f"py{j + 1},{f}"],
                            -variables[f"py{i + 1},{f + j_height}"]])

    for i in range(n):
        for j in range(i + 1, n):
            # lri,j ∨ lrj,i ∨ udi,j ∨ udj,i
            #Large-rectangles horizontal
            # cnf.append([-variables[f"a{i + 1}"], -variables[f"a{j + 1}"]])
            if min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > width:
                non_overlapping(False, i, j, False, False, True, True)
                non_overlapping(True, i, j, False, False, True, True)
            # Large rectangles vertical
            elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > height:
                non_overlapping(False, i, j, True, True, False, False)
                non_overlapping(True, i, j, True, True, False, False)

            # Same rectangle and is a square
            elif rectangles[i] == rectangles[j]:
                if rectangles[i][0] == rectangles[i][1]:
                    cnf.append([-variables[f"r{i + 1}"]])
                    cnf.append([-variables[f"r{j + 1}"]])
                    non_overlapping(False,i ,j, True, True, True, True)
                else:
                    non_overlapping(False, i, j, True, True, True, True)
                    non_overlapping(True, i, j, True, True, True, True)
            # normal rectangles
            else:
                non_overlapping(False, i, j, True, True, True, True)
                non_overlapping(True, i, j, True, True, True, True)


   # Domain encoding to ensure every rectangle stays inside strip's boundary
    for i in range(n):
        # cnf.append([-variables[f"a{i + 1}"]])
        if rectangles[i][0] > width: #if rectangle[i]'s width larger than strip's width, it has to be rotated
            cnf.append([variables[f"r{i + 1}"], -variables[f"a{i + 1}"]])
        else:
            for e in range(width - rectangles[i][0], width):
                    cnf.append([variables[f"r{i + 1}"],
                                variables[f"px{i + 1},{e}"], -variables[f"a{i + 1}"]])
        if rectangles[i][1] > height:
            cnf.append([variables[f"r{i + 1}"], -variables[f"a{i + 1}"]])
        else:
            for f in range(height - rectangles[i][1], height):
                    cnf.append([variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{f}"], -variables[f"a{i + 1}"]])

        # Rotated
        if rectangles[i][1] > width:
            cnf.append([-variables[f"r{i + 1}"], -variables[f"a{i + 1}"]])
        else:
            for e in range(width - rectangles[i][1], width):
                    cnf.append([-variables[f"r{i + 1}"],
                                variables[f"px{i + 1},{e}"], -variables[f"a{i + 1}"]])
        if rectangles[i][0] > height:
            cnf.append([-variables[f"r{i + 1}"], -variables[f"a{i + 1}"]])
        else:
            for f in range(height - rectangles[i][0], height):
                cnf.append([-variables[f"r{i + 1}"],
                            variables[f"py{i + 1},{f}"], -variables[f"a{i + 1}"]])
                
    # WCNF weight constraint of MAXSAT
    for i in range(n):
        cnf.append([variables[f"a{i + 1}"]], weight=profits[i])

    # print("Result of cnf", cnf)
    # output_file = f"{file_name}.wcnf"
    # cnf.to_file(output_file)
    # Soft clauses: a_i selected gives profit
    for i in range(n):
        cnf.append([variables[f"a{i+1}"]], weight=profits[i])

    output_file = f"{file_name}.wcnf"

    # ✅ Export to NEW WCNF format
    with open(output_file, "w") as f:
        f.write(f"c WCNF generated from {file_name}\n")
        f.write(f"c Format: h soft-literals (0 at the end)\n")
        soft_literals = [clause[0] for clause, weight in zip(cnf.soft, cnf.wght) if weight > 0]
        f.write("h " + " ".join(str(lit) for lit in soft_literals) + " 0\n")
        for clause in cnf.hard:
            f.write(" ".join(str(lit) for lit in clause) + " 0\n")

def max_profit_solution():
    folder_path = './dataset/soft_wbo'
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt') and file_name == "gcut8.txt":
            print(f"Processing file: {file_name}")
            with open(file_path, 'r') as file:
                lines = file.readlines()
                list_strip = list(map(int, lines[0].strip().split()))
                widths = list(map(int, lines[1].strip().split()))
                heights = list(map(int, lines[2].strip().split()))
                profits = list(map(int, lines[3].strip().split()))
                weights = list(map(int, lines[3].strip().split()))

                rectangles = [(widths[i], heights[i], profits[i], weights[i]) for i in range(len(widths))]
                start_time = time.time()
                maxsat_constraints(rectangles, list_strip, profits, file_name, weights)
                print(f"{file_name} done in {time.time() - start_time:.2f}s")

max_profit_solution()

