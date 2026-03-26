import random
import time

from matmult import matmul
from strassen import strassen

import matplotlib.pyplot as plt

SIZES = [32, 64, 128, 256]
REPEATS_PER_SIZE = 4
CANDIDATE_CROSSOVERS = [28+4*x for x in range(25)]
SEED = 0
OUTPUT_PATH = "benchmark_crossover_search01.png"


def make_matrix(size, rng):
    return [[rng.choice([0, 1]) for _ in range(size)] for _ in range(size)]


def build_cases():
    rng = random.Random(SEED)
    cases = []
    for size in SIZES:
        for _ in range(REPEATS_PER_SIZE):
            A = make_matrix(size, rng)
            B = make_matrix(size, rng)
            cases.append((size, A, B))
    return cases


def timed_run(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return end - start, result


def benchmark_matmul(cases):
    timings = {size: [] for size in SIZES}

    for size, A, B in cases:
        runtime, _ = timed_run(matmul, A, B)
        timings[size].append(runtime)

    return {size: sum(values) / len(values) for size, values in timings.items()}


def benchmark_crossovers(cases):
    results = []

    for crossover in CANDIDATE_CROSSOVERS:
        timings = {size: [] for size in SIZES}
        total_time = 0.0

        for size, A, B in cases:
            runtime, strassen_result = timed_run(strassen, A, B, crossover)
            expected = matmul(A, B)
            if strassen_result != expected:
                raise ValueError(
                    f"Results did not match for size {size} at crossover {crossover}"
                )

            timings[size].append(runtime)
            total_time += runtime

        average_by_size = {
            size: sum(values) / len(values) for size, values in timings.items()
        }

        results.append(
            {
                "crossover": crossover,
                "total_time": total_time,
                "average_by_size": average_by_size,
            }
        )

        size_summary = "  ".join(
            f"n={size}: {average_by_size[size]:.6f}s" for size in SIZES
        )
        print(
            f"crossover={crossover:2d}  total={total_time:.6f}s  {size_summary}"
        )

    return results


def plot_results(results, matmul_baseline):

    crossovers = [row["crossover"] for row in results]
    totals = [row["total_time"] for row in results]

    plt.figure(figsize=(9, 5))
    plt.plot(crossovers, totals, marker="o", label="Strassen total runtime")
    plt.title("Finding the Best Strassen Crossover")
    plt.xlabel("Crossover size")
    plt.ylabel("Total runtime across benchmark cases (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Wrote graph to {OUTPUT_PATH}")

    print("Matmul baseline by size:")
    for size in SIZES:
        print(f"  n={size}: {matmul_baseline[size]:.6f}s")


def main():
    cases = build_cases()
    matmul_baseline = benchmark_matmul(cases)
    results = benchmark_crossovers(cases)

    if not results:
        print("No crossover benchmarks were run.")
        return

    best = min(results, key=lambda row: row["total_time"])
    print()
    print(f"Best crossover: {best['crossover']}")
    print(f"Best total runtime: {best['total_time']:.6f}s")
    for size in SIZES:
        print(f"  n={size}: {best['average_by_size'][size]:.6f}s")

    plot_results(results, matmul_baseline)


if __name__ == "__main__":
    main()
