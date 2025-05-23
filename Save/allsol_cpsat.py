from ortools.sat.python import cp_model

class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, x_vars, y_vars, rectangles):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._x_vars = x_vars
        self._y_vars = y_vars
        self._rectangles = rectangles

        self._solutions = []

    def on_solution_callback(self):
        solution = []
        for i in range(len(self._rectangles)):
            x_val = self.Value(self._x_vars[i])
            y_val = self.Value(self._y_vars[i])
            solution.append((x_val, y_val))
        self._solutions.append(solution)
        print(f'Solution {len(self._solutions)}:')
        for i, (x, y) in enumerate(solution):
            print(f'  Rectangle {i} (width: {self._rectangles[i][0]}, height: {self._rectangles[i][1]}): bottom-left corner at ({x}, {y})')

    def get_solutions(self):
        return self._solutions

def orthogonal_packing_problem(rectangles, W, H):
    model = cp_model.CpModel()

    n = len(rectangles)
    
    x = [model.NewIntVar(0, W, f'x_{i}') for i in range(n)]
    y = [model.NewIntVar(0, H, f'y_{i}') for i in range(n)]

    for i in range(n):
        wi, hi = rectangles[i]
        model.Add(x[i] + wi <= W)  
        model.Add(y[i] + hi <= H) 

    for i in range(n):
        for j in range(i + 1, n):
            wi, hi = rectangles[i]
            wj, hj = rectangles[j]

            no_overlap_1 = model.NewBoolVar(f'no_overlap_1_{i}_{j}')  # x[i] + wi <= x[j]
            no_overlap_2 = model.NewBoolVar(f'no_overlap_2_{i}_{j}')  # x[j] + wj <= x[i]
            no_overlap_3 = model.NewBoolVar(f'no_overlap_3_{i}_{j}')  # y[i] + hi <= y[j]
            no_overlap_4 = model.NewBoolVar(f'no_overlap_4_{i}_{j}')  # y[j] + hj <= y[i]

            model.Add(x[i] + wi <= x[j]).OnlyEnforceIf(no_overlap_1)
            model.Add(x[j] + wj <= x[i]).OnlyEnforceIf(no_overlap_2)
            model.Add(y[i] + hi <= y[j]).OnlyEnforceIf(no_overlap_3)
            model.Add(y[j] + hj <= y[i]).OnlyEnforceIf(no_overlap_4)

            model.AddBoolOr([no_overlap_1, no_overlap_2, no_overlap_3, no_overlap_4])

    solver = cp_model.CpSolver()

    # Callback to find all solutions
    solution_collector = SolutionCollector(x, y, rectangles)

    status = solver.SearchForAllSolutions(model, solution_collector)

    print(f'Number of solutions found: {len(solution_collector.get_solutions())}')

rectangles = [(1, 2), (1, 2), (2, 1), (1, 1)]
W, H = 4, 2  # Width and height of the strip
orthogonal_packing_problem(rectangles, W, H)