from matmult import matmul

CROSSOVER_SIZE = 68


def validateMatrix(m):
    if not isinstance(m, list) or not all(isinstance(row, list) for row in m):
        raise TypeError("Matrix needs a list of lists")
    if not m:
        raise ValueError("Matrix dimension must be positive")

    row_len = len(m[0])
    if row_len == 0:
        raise ValueError("Matrix dimension must be positive")
    if any(len(row) != row_len for row in m):
        raise ValueError("All rows must have the same length")


def matrix_add(X, Y):
    n = len(X)
    return [[X[i][j] + Y[i][j] for j in range(n)] for i in range(n)]


def matrix_sub(X, Y):
    n = len(X)
    return [[X[i][j] - Y[i][j] for j in range(n)] for i in range(n)]


def add(X, Y):
    if len(X) != len(Y):
        raise ValueError("Must have same dimensions")
    return matrix_add(X, Y)


def sub(X, Y):
    if len(X) != len(Y):
        raise ValueError("Must have same dimensions")
    return matrix_sub(X, Y)


def join_quadrants(A, B, C, D):
    top = [row_a + row_b for row_a, row_b in zip(A, B)]
    bottom = [row_c + row_d for row_c, row_d in zip(C, D)]
    return top + bottom


def split(m):
    n = len(m)
    if n & (n - 1) != 0:
        raise ValueError("Need power of two")

    half = n // 2
    A = [row[:half] for row in m[:half]]
    B = [row[half:] for row in m[:half]]
    C = [row[:half] for row in m[half:]]
    D = [row[half:] for row in m[half:]]
    return A, B, C, D


def pad_matrix(m, size):
    padded = [[0 for _ in range(size)] for _ in range(size)]
    for i, row in enumerate(m):
        padded[i][: len(row)] = row[:]
    return padded


def trim_matrix(mat, size):
    return [row[:size] for row in mat[:size]]


def next_power_two(n):
    power = 1
    while power < n:
        power *= 2
    return power


def strassen(X, Y, crossover_size=None):
    if crossover_size is None:
        crossover_size = CROSSOVER_SIZE

    n = len(X)
    if n != len(Y):
        raise ValueError("Must have same dimensions")

    pad_len = next_power_two(n)
    padX = pad_matrix(X, pad_len)
    padY = pad_matrix(Y, pad_len)

    return trim_matrix(strassen_recurse(padX, padY, crossover_size), n)


def strassen_recurse(X, Y, crossover_size):
    n = len(X)
    if n <= crossover_size:
        return matmul(X, Y)

    A, B, C, D = split(X)
    E, F, G, H = split(Y)

    P1 = strassen_recurse(A, matrix_sub(F, H), crossover_size)
    P2 = strassen_recurse(matrix_add(A, B), H, crossover_size)
    P3 = strassen_recurse(matrix_add(C, D), E, crossover_size)
    P4 = strassen_recurse(D, matrix_sub(G, E), crossover_size)
    P5 = strassen_recurse(matrix_add(A, D), matrix_add(E, H), crossover_size)
    P6 = strassen_recurse(matrix_sub(B, D), matrix_add(G, H), crossover_size)
    P7 = strassen_recurse(matrix_sub(C, A), matrix_add(E, F), crossover_size)

    C11 = matrix_add(matrix_sub(matrix_add(P5, P4), P2), P6)
    C12 = matrix_add(P1, P2)
    C21 = matrix_add(P3, P4)
    C22 = matrix_add(matrix_sub(matrix_add(P1, P5), P3), P7)

    return join_quadrants(C11, C12, C21, C22)



