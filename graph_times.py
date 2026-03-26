import random
import time

from matmult import matmul
from strassen import strassen

import matplotlib.pyplot as plt

SWITCH = 128

MIN_SIZE = 2
MAX_SIZE = 512
REPEATS = 3
CUTOFF_SECONDS = 100
SEED = 0
OUTPUT_PATH = f"runtime_comparison_threshold{SWITCH}.png"


def make_matrix(size, rng):
    return [[rng.choice([-1, 0, 1]) for _ in range(size)] for _ in range(size)]


def timed_run(func, A, B):
    start = time.perf_counter()
    result = func(A, B)
    end = time.perf_counter()
    return end - start, result


def choose_sizes():
    sizes = []
    size = MIN_SIZE
    while size <= MAX_SIZE:
        sizes.append(size)
        size *= 2
    return sizes


def benchmark():
    rng = random.Random(SEED)
    results = []

    for size in choose_sizes():
        matmul_times = []
        strassen_times = []

        for _ in range(REPEATS):
            A = make_matrix(size, rng)
            B = make_matrix(size, rng)

            matmul_time, matmul_result = timed_run(matmul, A, B)
            strassen_time, strassen_result = timed_run(strassen, A, B)

            if matmul_result != strassen_result:
                raise ValueError(f"Results did not match for size {size}")

            matmul_times.append(matmul_time)
            strassen_times.append(strassen_time)

        avg_matmul = sum(matmul_times) / len(matmul_times)
        avg_strassen = sum(strassen_times) / len(strassen_times)
        results.append((size, avg_matmul, avg_strassen))

        print(
            f"n={size:4d}  matmul={avg_matmul:.6f}s  strassen={avg_strassen:.6f}s"
        )

        if avg_matmul > CUTOFF_SECONDS or avg_strassen > CUTOFF_SECONDS:
            print(
                f"Stopping early because the cutoff of {CUTOFF_SECONDS:.2f}s was exceeded."
            )
            break

    return results


def plot_results(results):
    sizes = [row[0] for row in results]
    matmul_times = [row[1] for row in results]
    strassen_times = [row[2] for row in results]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, matmul_times, marker="o", label="matmul")
    plt.plot(sizes, strassen_times, marker="o", label="strassen")
    plt.title("Matrix Multiplication Runtime Comparison")
    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Wrote graph to {OUTPUT_PATH}")


def main():
    results = benchmark()
    if not results:
        print("No benchmarks were run.")
        return
    plot_results(results)


if __name__ == "__main__":
    main()
