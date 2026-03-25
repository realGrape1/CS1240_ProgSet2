from matmult import matmul

CROSSOVER_SIZE = 32


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


class MatrixView:
    def __init__(self, m, i1, i2, j1, j2):
        validateMatrix(m)
        if i1 < 0 or j1 < 0 or i2 > len(m) or j2 > len(m[0]):
            raise IndexError("MatrixView bounds out of range")
        if i2 - i1 != j2 - j1:
            raise ValueError("MatrixView must be square")

        self.matrix = m
        self.i1 = i1
        self.i2 = i2
        self.j1 = j1
        self.j2 = j2
        self.length = i2 - i1

    def __repr__(self):
        lines = []
        for i in range(self.length):
            lines.append(
                ",".join(str(self.matrix[self.i1 + i][self.j1 + j]) for j in range(self.length))
            )
        return "[\n" + "\n".join(lines) + "\n]"

    def to_list(self):
        return [
            [self.matrix[self.i1 + i][self.j1 + j] for j in range(self.length)]
            for i in range(self.length)
        ]


def as_matrix(m):
    if isinstance(m, MatrixView):
        m = m.to_list()

    validateMatrix(m)
    if len(m) != len(m[0]):
        raise ValueError("Must be square")
    return m


def matrix_add(X, Y):
    n = len(X)
    return [[X[i][j] + Y[i][j] for j in range(n)] for i in range(n)]


def matrix_sub(X, Y):
    n = len(X)
    return [[X[i][j] - Y[i][j] for j in range(n)] for i in range(n)]


def add(X, Y):
    X = as_matrix(X)
    Y = as_matrix(Y)
    if len(X) != len(Y):
        raise ValueError("Must have same dimensions")
    return matrix_add(X, Y)


def sub(X, Y):
    X = as_matrix(X)
    Y = as_matrix(Y)
    if len(X) != len(Y):
        raise ValueError("Must have same dimensions")
    return matrix_sub(X, Y)


def join_quadrants(A, B, C, D):
    top = [row_a + row_b for row_a, row_b in zip(A, B)]
    bottom = [row_c + row_d for row_c, row_d in zip(C, D)]
    return top + bottom


def split(m):
    m = as_matrix(m)
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


def strassen(X, Y):
    X = as_matrix(X)
    Y = as_matrix(Y)

    n = len(X)
    if n != len(Y):
        raise ValueError("Must have same dimensions")

    pad_len = next_power_two(n)
    padX = pad_matrix(X, pad_len)
    padY = pad_matrix(Y, pad_len)

    return trim_matrix(strassen_recurse(padX, padY), n)


def strassen_recurse(X, Y):
    n = len(X)
    if n <= CROSSOVER_SIZE:
        return matmul(X, Y)

    A, B, C, D = split(X)
    E, F, G, H = split(Y)

    P1 = strassen_recurse(A, matrix_sub(F, H))
    P2 = strassen_recurse(matrix_add(A, B), H)
    P3 = strassen_recurse(matrix_add(C, D), E)
    P4 = strassen_recurse(D, matrix_sub(G, E))
    P5 = strassen_recurse(matrix_add(A, D), matrix_add(E, H))
    P6 = strassen_recurse(matrix_sub(B, D), matrix_add(G, H))
    P7 = strassen_recurse(matrix_sub(C, A), matrix_add(E, F))

    C11 = matrix_add(matrix_sub(matrix_add(P5, P4), P2), P6)
    C12 = matrix_add(P1, P2)
    C21 = matrix_add(P3, P4)
    C22 = matrix_add(matrix_sub(matrix_add(P1, P5), P3), P7)

    return join_quadrants(C11, C12, C21, C22)



