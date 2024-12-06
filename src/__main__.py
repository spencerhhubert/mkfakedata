from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from entropy import *
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, shape: Tuple[int, ...]):
        self.grid = np.zeros(shape)
        self.position = tuple(np.random.randint(0, s) for s in shape)

    def movePosition(self, deltas: List[float]):
        new_pos = []
        for i, (p, d, s) in enumerate(zip(self.position, deltas, self.grid.shape)):
            new_p = int(p + d * s)  # Scale movement by dimension size
            new_pos.append(min(max(0, new_p), s - 1))
        self.position = tuple(new_pos)


class GridOpParams:
    def __init__(self, grid: Grid):
        self.grid = grid


class GridOp:
    def __init__(self, func: Callable[..., None], param_count: int):
        self.func = func
        self.param_count = param_count


class GridOpCall:
    def __init__(self, op_idx: int, params: List[float]):
        self.op_idx = op_idx
        self.params = params


def propagateFromPoint(params: GridOpParams, strength: float, value: float) -> None:
    indices = np.indices(params.grid.grid.shape)
    squared_diffs = sum(
        (ind - pos) ** 2 for ind, pos in zip(indices, params.grid.position)
    )
    distances = np.sqrt(squared_diffs)
    mask = distances < strength
    params.grid.grid[mask] = value


def moveRelative(params: GridOpParams, *deltas: float) -> None:
    params.grid.movePosition(list(deltas))


shape = (10, 10)  # Default shape
target_entropy = 3  # Default target entropy

GridOps = [GridOp(propagateFromPoint, 2), GridOp(moveRelative, len(shape))]


def decodeGenesToOperations(genes: List[float]) -> List[GridOpCall]:
    ops = []
    i = 0
    while i < len(genes):
        op_idx = int(genes[i] * len(GridOps))
        op_idx = max(0, min(op_idx, len(GridOps) - 1))
        i += 1

        num_params = GridOps[op_idx].param_count
        if i + num_params > len(genes):
            break
        params = genes[i : i + num_params]
        i += num_params

        ops.append(GridOpCall(op_idx, params))
    return ops


def applyOperation(grid: Grid, op_call: GridOpCall) -> None:
    op = GridOps[op_call.op_idx]
    params = GridOpParams(grid)
    op.func(params, *op_call.params)


def applyOperationSequence(
    shape: Tuple[int, ...], operations: List[GridOpCall]
) -> np.ndarray:
    grid = Grid(shape)
    for op_call in operations:
        applyOperation(grid, op_call)
    return grid.grid


def calcFitness(ga_instance, solution: list, solution_idx: int) -> float:
    operations = decodeGenesToOperations(solution)
    pattern = applyOperationSequence(shape, operations)
    current_entropy = calcShannonEntropy(pattern)

    # Penalize completely empty or full grids
    emptiness_penalty = -2.0 if np.all(pattern == 0) or np.all(pattern == 1) else 0.0

    # Base score starts positive and decreases with distance from target
    # This creates a gradient even when far from the target
    base_score = 10.0 - abs(target_entropy - current_entropy)

    return base_score + emptiness_penalty


def displayGrid(grid: np.ndarray) -> None:
    if grid.ndim == 2:
        plt.figure(figsize=(8, 8))

        # Use min/max of data for color scaling
        vmin, vmax = np.min(grid), np.max(grid)

        # If all values are the same, adjust range to prevent division by zero
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        plt.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Value")

        # Add grid lines
        plt.grid(True, which="major", color="black", linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])

        plt.title(f"Grid Values: min={vmin:.2f}, max={vmax:.2f}")
        plt.show()
    else:
        print("Grid shape:", grid.shape)


def runGeneticAlgorithm(
    num_genes: int = 100, num_generations: int = 200, num_solutions: int = 100
) -> Tuple[np.ndarray, float]:
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=8,
        num_genes=num_genes,
        init_range_low=0.0,
        init_range_high=1.0,
        fitness_func=calcFitness,
        sol_per_pop=num_solutions,
        gene_type=float,
        mutation_percent_genes=20,
        mutation_type="random",
        crossover_type="single_point",
        keep_parents=4,
    )

    ga_instance.run()

    print(f"Generation reached: {ga_instance.generations_completed}")
    print(f"Best fitness achieved: {ga_instance.best_solution()[1]}")

    solution, solution_fitness, _ = ga_instance.best_solution()
    operations = decodeGenesToOperations(solution)
    best_pattern = applyOperationSequence(shape, operations)

    return best_pattern, solution_fitness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arc", dest="arc_data_path", required=False, help="Path to ARC data"
    )
    args = parser.parse_args()

    pattern, fitness = runGeneticAlgorithm()
    print(f"Best fitness: {fitness}")
    displayGrid(pattern)


if __name__ == "__main__":
    main()
