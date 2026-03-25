import pytest

from matmult import matmul
from strassen import strassen


def naive_matmul(A, B):
    rows = len(A)
    inner = len(B)
    cols = len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


@pytest.mark.parametrize("multiply", [matmul, strassen])
def test_1x1_multiplication(multiply):
    assert multiply([[7]], [[-3]]) == [[-21]]


@pytest.mark.parametrize("multiply", [matmul, strassen])
def test_2x2_multiplication(multiply):
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert multiply(A, B) == [[19, 22], [43, 50]]


@pytest.mark.parametrize("multiply", [matmul, strassen])
def test_4x4_matches_naive(multiply):
    A = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    B = [
        [2, 0, 1, 3],
        [1, 2, 0, 1],
        [3, 1, 2, 0],
        [0, 4, 1, 2],
    ]
    assert multiply(A, B) == naive_matmul(A, B)


@pytest.mark.parametrize("multiply", [matmul, strassen])
def test_negative_entries(multiply):
    A = [[2, -1], [0, 3]]
    B = [[4, 5], [-2, 1]]
    assert multiply(A, B) == naive_matmul(A, B)


@pytest.mark.parametrize("multiply", [matmul, strassen])
def test_dimension_mismatch_raises(multiply):
    A = [[1, 2], [3, 4]]
    B = [[1]]
    with pytest.raises(ValueError):
        multiply(A, B)


@pytest.mark.parametrize("multiply", [matmul, strassen])
def test_non_square_input_raises(multiply):
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        multiply(A, B)


def test_matmul_3x3_square_input():
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    B = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1],
    ]
    assert matmul(A, B) == naive_matmul(A, B)


def test_strassen_3x3_square_input():
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    B = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1],
    ]
    assert strassen(A, B) == naive_matmul(A, B)
