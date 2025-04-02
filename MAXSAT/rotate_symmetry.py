import os
import time
from utils.visualization import display_solution
from utils.export import export_csv
from solver.maxsat_solver import maxsat_constraints

def max_profit_solution():
    folder_path = './dataset/'
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            print(f"Processing file: {file_name}")
            with open(file_path, 'r') as file:
                lines = file.readlines()
                list_strip = list(map(int, lines[0].strip().split()))
                widths = list(map(int, lines[1].strip().split()))
                heights = list(map(int, lines[2].strip().split()))
                profits = list(map(int, lines[3].strip().split()))
                num_items = len(widths)
                
                # Create rectangle list
                rectangles = []
                for j in range(num_items):
                    item = (widths[j], heights[j], profits[j])
                    rectangles.append(item)

                # Solve the problem
                started_time = time.time()
                status = maxsat_constraints(rectangles, list_strip, profits, file_name)
                ended_time = time.time()
                
                print(f"{file_name}: {ended_time - started_time:.2f} seconds")
                
                # Export results
                if status[0] == "TIMEOUT":
                    export_csv(file_name, "maxsat_rotate_symmetry", ended_time - started_time, "TIMEOUT", 0, 0, 0)
                elif status[0] == "SAT":
                    export_csv(file_name, "maxsat_rotate_symmetry", ended_time - started_time, "SAT", status[1], status[2], status[3])
                else:  # UNSAT
                    export_csv(file_name, "maxsat_rotate_symmetry", ended_time - started_time, "UNSAT", 0, 0, 0)

if __name__ == "__main__":
    max_profit_solution()