import time

def matmul(A, B):
    if len(A) != len(A[0]):
        raise ValueError("Must be square")
    if len(B) != len(B[0]):
        raise ValueError("Must be square")

    if len(A) != len(B):
        raise ValueError("Must have same dimensions")
    
    n = len(A)

    # Answer matrix
    C = [[0 for j in range(n)] for i in range(n)]

    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik*Bk[j]

    return C

N = 100
A = [[2 for i in range(N)] for j in range(N)]
B = [[2 for i in range(N)] for j in range(N)]

start_time = time.perf_counter()
result = matmul(A,B)
end_time = time.perf_counter()


print("TIME: ", end_time-start_time)

