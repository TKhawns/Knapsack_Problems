from ortools.sat.python import cp_model

def cpSAT_OPP(rectangles, W, H):
    model = cp_model.CpModel()

    n = len(rectangles)
    
    # Variable encoding (x[i], y[i]) = (w[i], h[i])
    x = [model.NewIntVar(0, W, f'x_{i}') for i in range(n)]
    y = [model.NewIntVar(0, H, f'y_{i}') for i in range(n)]

    # (1) Domain of rectangle (x[i], y[i])
    for i in range(n):
        wi, hi = rectangles[i]
        print(wi, hi)
        model.Add(x[i] + wi <= W) 
        model.Add(y[i] + hi <= H)

    #(2) Non-overlapping constraints
    for i in range(n):
        for j in range(i + 1, n):
            wi, hi = rectangles[i]
            wj, hj = rectangles[j]

            # Boolean variables
            no_overlap_1 = model.NewBoolVar(f'nonoverlap_1_{i}_{j}')  # x[i] + wi <= x[j]
            no_overlap_2 = model.NewBoolVar(f'nonoverlap_2_{i}_{j}')  # x[j] + wj <= x[i]
            no_overlap_3 = model.NewBoolVar(f'nonoverlap_3_{i}_{j}')  # y[i] + hi <= y[j]
            no_overlap_4 = model.NewBoolVar(f'nonoverlap_4_{i}_{j}')  # y[j] + hj <= y[i]

            model.Add(x[i] + wi <= x[j]).OnlyEnforceIf(no_overlap_1)
            model.Add(x[j] + wj <= x[i]).OnlyEnforceIf(no_overlap_2)
            model.Add(y[i] + hi <= y[j]).OnlyEnforceIf(no_overlap_3)
            model.Add(y[j] + hj <= y[i]).OnlyEnforceIf(no_overlap_4)

            # At least one constraints
            model.AddBoolOr([no_overlap_1, no_overlap_2, no_overlap_3, no_overlap_4])

    # CP Solver
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    print(len(model.Proto().variables))

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        print('Solution found:')
        for i in range(n):
            print(f'Rectangle {i}: ({solver.Value(x[i])}, {solver.Value(y[i])})')
    else:
        print('No solution found.')

# List (width, height)
rectangles = [(1, 2), (1, 2), (2, 1), (1, 1)]  
# Strip (W, H)
W, H = 4, 2  
cpSAT_OPP(rectangles, W, H)
