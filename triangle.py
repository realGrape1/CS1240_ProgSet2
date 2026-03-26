import random
import time

import matplotlib.pyplot as plt

from strassen import strassen


NUM_VERTICES = 1024
EDGE_PROBABILITIES = [0.01, 0.02, 0.03, 0.04, 0.05]
SEED = 0
OUTPUT_PATH = "triangle_counts.png"


def generate_random_graph(n, p, rng):
    adjacency = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            edge = 1 if rng.random() < p else 0
            adjacency[i][j] = edge
            adjacency[j][i] = edge

    return adjacency


def count_triangles(adjacency):
    squared = strassen(adjacency, adjacency)
    cubed = strassen(squared, adjacency)
    diagonal_sum = sum(cubed[i][i] for i in range(len(cubed)))
    return diagonal_sum // 6


def expected_triangles(n, p):
    return n*(n-1)*(n-2)//6 * (p ** 3)


def run_experiment():
    rng = random.Random(SEED)
    results = []

    for p in EDGE_PROBABILITIES:
        graph = generate_random_graph(NUM_VERTICES, p, rng)

        start = time.perf_counter()
        triangles = count_triangles(graph)
        end = time.perf_counter()

        result = {
            "p": p,
            "observed": triangles,
            "expected": expected_triangles(NUM_VERTICES, p),
            "runtime_seconds": end - start,
        }
        results.append(result)

        print(
            f"p={p:.2f}  observed={triangles}  "
            f"expected={result['expected']:.2f}  "
            f"time={result['runtime_seconds']:.2f}s"
        )

    return results


def plot_results(results):
    probabilities = [row["p"] for row in results]
    observed = [row["observed"] for row in results]
    expected = [row["expected"] for row in results]

    plt.figure(figsize=(9, 5))
    plt.plot(probabilities, observed, marker="o", linewidth=2, label="Observed")
    plt.plot(
        probabilities,
        expected,
        marker="o",
        linestyle="--",
        label="Expected",
    )
    plt.title("Triangles in Random Graphs on 1024 Vertices")
    plt.xlabel("Edge probability p")
    plt.ylabel("Number of triangles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Wrote chart to {OUTPUT_PATH}")


def main():
    results = run_experiment()
    plot_results(results)


if __name__ == "__main__":
    main()
