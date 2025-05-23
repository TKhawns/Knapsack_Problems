from pysat.solvers import Glucose3

#  at least one function
# A = x1 v x2 v x3 v x4 v ... v xn
# return: "x1 x2 x3 ... xn 0"
def atLeastOne(A):
    temp = ""
    for x in A:
        temp += " " + str(x)
    temp += " 0\n"
    return temp

# at most one function
# A = x1,x2,x3,...,xn
# return: (-x1 v -x2) ^ (-x1 v -x3) ^ ... ^ (-x1 v -xn)
# return: (-x2 v -x3) ^ (-x2 v -x3) ^ ... ^ (-x2 v -xn)
# return: ...
def atMostOne(A):
    temp = ""
    for x in A:
        for y in A[A.index(x)+1:]:
            temp += " -" + str(x) + " -" + str(y) + " 0\n"
    return temp

# exactly one function = at least + at most
def exactlyOne(A):
    temp = ""
    temp += atLeastOne(A) + atMostOne(A)
    return temp


# mapping position (i,j) to order integer from 1.
def mapPosition(row, col, N):
    return row * N + col + 1

# read input
N = 4
# check for same input
if N < 1:
    print("Error N < 1")
    sys.exit(0)

# Start Solver
print("c SAT Expression for N = " + str(N))
spots = N*N
print("c Board has " + str(spots) + " positions")

# Exactly 1 queen per row
temp = ""
for row in range(0,N):
    A = []
    for column in range(0,N):
        position = mapPosition(row,column,N)
        A.append(position)
    temp = temp + exactlyOne(A)

# Exactly 1 queen per column
for column in range(0,N):
    A = []
    for row in range(0,N):
        position = mapPosition(row,column,N)
        A.append(position)
    temp = temp + exactlyOne(A)

# At most 1 queen per negative diagonal from left
for row in range(N-1,-1,-1):
    A = []
    for x in range(0,N-row):
        A.append(mapPosition(row+x,x,N))
    temp = temp + atMostOne(A)

# At most 1 queen per negative diagonal from top
for column in range(1,N):
    A = []
    for x in range(0,N-column):
        A.append(mapPosition(x,column+x,N))
    temp = temp + atMostOne(A)

# At most 1 queen per positive diagonal from right
for row in range(N-1,-1,-1):
    A = []
    for x in range(0,N-row):
        A.append(mapPosition(row+x,N-1-x,N))
    temp = temp + atMostOne(A)

# At most 1 queen per positive diagonal from top
for column in range(N-2,-1,-1):
    A = []
    for x in range(0,column+1):
        A.append(mapPosition(x,column-x,N))
    temp = temp + atMostOne(A)

print('p cnf ' + str(N*N) + ' ' + str(temp.count('\n')) + '\n')
print(temp)

# Convert string to 2D array, input of mini-SAT Glucose3
rows = temp.strip().split("\n")
result = []
for row in rows:
    numbers_as_strings = row.strip().split()
    # Convert number strings to integers, remove element "0"
    numbers = [int(num_str) for num_str in numbers_as_strings[:-1]] 
    result.append(numbers)

def solution():
    solver = Glucose3()

    for clause in result:
        solver.add_clause(clause)
    
    if solver.solve():
        model = solver.get_model()
        print("Model")
        print(model)
        return [[int(model[i * N + j] > 0) for j in range(N)] for i in range(N)]
    else:
        return None
    
def print_solution(solution):
    if solution is None:
        print("No solution found.")
    else:
        for row in solution:
            print(" ".join("Q" if cell else "." for cell in row))
print(solution())
print_solution(solution())