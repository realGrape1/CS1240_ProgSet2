def validateMatrix(m):
    if not isinstance(m, list) or not all(isinstance(row, list) for row in m):
        raise TypeError("Matrix needs a list of lists")
    if m:
        row_len = len(m[0])
        if any(len(row) != row_len for row in m):
            raise ValueError("All rows must have the same length")


class MatrixView:
    def __init__(self, m, i1, i2, j1, j2):
        validateMatrix(m)
        if i1 < 0 or j1 < 0 or i2 > len(m) or (m and j2 > len(m[0])):
            raise IndexError("MatrixView bounds out of range")
        if i2 - i1 != j2 - j1:
            raise ValueError("MatrixView must be square")

        self.matrix = m
        self.i1 = i1
        self.i2 = i2
        self.j1 = j1
        self.j2 = j2
        self.length = i2 - i1

    # Just to help print
    def __repr__(self):
        lines = []
        for i in range(self.length):
            lines.append(
                ",".join(str(self.value_at(i, j)) for j in range(self.length))
            )
        return "[\n" + "\n".join(lines) + "\n]"

    def value_at(self, i, j):
        return self.matrix[self.i1 + i][self.j1 + j]

    # Subview like this so we can conveniently split into 4
    def subview(self, i1, i2, j1, j2):
        return MatrixView(
            self.matrix,
            self.i1 + i1,
            self.i1 + i2,
            self.j1 + j1,
            self.j1 + j2,
        )

    def to_list(self):
        return [[self.value_at(i, j) for j in range(self.length)] 
                for i in range(self.length)]


def as_view(m) -> MatrixView:
    if isinstance(m, MatrixView):
        return m

    validateMatrix(m)
    if len(m) != len(m[0]):
        raise ValueError("Must be square")
    
    return MatrixView(m, 0, len(m), 0, len(m))


def matrix_add(X, Y):
    X = as_view(X)
    Y = as_view(Y)
    if X.length != Y.length:
        raise ValueError("Must have same dimensions")

    return [[X.value_at(i, j) + Y.value_at(i, j) for j in range(X.length)]
        for i in range(X.length)]


def matrix_sub(X, Y):
    X = as_view(X)
    Y = as_view(Y)
    if X.length != Y.length:
        raise ValueError("Must have the same dimensions")

    return [[X.value_at(i, j) - Y.value_at(i, j) for j in range(X.length)]
        for i in range(X.length)]


def add(X, Y):
    return matrix_add(X, Y)

def sub(X, Y):
    return matrix_sub(X, Y)


def join_quadrants(A, B, C, D):
    """
    Join four list[lists], NOT MatrixViews
    """
    top = [row_a + row_b for row_a, row_b in zip(A, B)]
    bottom = [row_c + row_d for row_c, row_d in zip(C, D)]
    return top + bottom


def split(mv):
    mv = as_view(mv)
    n = mv.length
    if n == 0 or n & (n - 1) != 0:
        raise ValueError("Need power of two")

    half = n // 2
    A = mv.subview(0, half, 0, half)
    B = mv.subview(0, half, half, n)
    C = mv.subview(half, n, 0, half)
    D = mv.subview(half, n, half, n)

    return A, B, C, D

# Padding logic
# ==========================
def pad_matrix(m, size):
    padded = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(len(m)):
        for j in range(len(m[i])):
            padded[i][j] = m[i][j]
    return padded

def trim_matrix(mat, size):
    return [row[:size] for row in mat[:size]]

def next_power_two(n):
    power = 1
    while power<n:
        power *= 2
    return power

# ==========================


# Wrapper for recursive strassen. Handles padding
def strassen(X, Y):
    X = as_view(X)
    Y = as_view(Y)

    n = X.length
    if n != Y.length:
        raise ValueError("Must have same dimensions")
    if n == 0:
        raise ValueError("Matrix dimension must be positive")

    pad_len = next_power_two(n)

    padX = pad_matrix(X.to_list(), pad_len)
    padY = pad_matrix(Y.to_list(), pad_len)

    return trim_matrix(strassen_recurse(padX, padY), n)


# Recursive strassen handles divide and combine
def strassen_recurse(X, Y):
    """
    Returns: list[list]
    """

    X = as_view(X)
    Y = as_view(Y)

    n = X.length
    if n != Y.length:
        raise ValueError("Must have same dimensions")
    if n == 0 :
        raise ValueError("Matrix dimension must be positive")

    if n == 1:
        return [[ X.value_at(0, 0) * Y.value_at(0, 0) ]]

    A, B, C, D = split(X)
    E, F, G, H = split(Y)

    P1 = strassen_recurse(A, sub(F, H))
    P2 = strassen_recurse(add(A, B), H)
    P3 = strassen_recurse(add(C, D), E)
    P4 = strassen_recurse(D, sub(G, E))
    P5 = strassen_recurse(add(A, D), add(E, H))
    P6 = strassen_recurse(sub(B, D), add(G, H))
    P7 = strassen_recurse(sub(C, A), add(E, F))

    C11 = add(sub(add(P5, P4), P2), P6)
    C12 = add(P1, P2)
    C21 = add(P3, P4)
    C22 = add(sub(add(P1, P5), P3), P7)

    return join_quadrants(C11, C12, C21, C22)


if __name__ == "__main__":
    test_matrix1 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]

    test_matrix2 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]

    for x in split(test_matrix1):
        print(x)

    print(strassen(test_matrix1, test_matrix2))
