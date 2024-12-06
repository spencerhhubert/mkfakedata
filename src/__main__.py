from entropy import *
from grid import *
from ops import *
from util import *
from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from tqdm import tqdm
import time


shape = (10, 10)
target_entropy = 7

GridOps = [
    GridOp(propagateFromPoint, 2),
    GridOp(moveRelative, len(shape)),
    GridOp(line, 3),
    GridOp(fill, 2),
    GridOp(remove, 1),
    GridOp(moduloFill, 3),
    GridOp(rectangle, 2),
    GridOp(repeatNthAgo, 2),
]


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
    # Update the progress bar for solutions if it exists
    if hasattr(ga_instance, "pbar_solutions"):
        ga_instance.pbar_solutions.update(1)

    operations = decodeGenesToOperations(solution)
    pattern = applyOperationSequence(shape, operations)
    current_entropy = calcShannonEntropy(pattern)

    emptiness_penalty = -2.0 if np.all(pattern == 0) or np.all(pattern == 1) else 0.0
    base_score = 10.0 - abs(target_entropy - current_entropy)

    return base_score + emptiness_penalty


def runGeneticAlgorithm(
    num_genes: int = 100, num_generations: int = 200, num_solutions: int = 100
) -> Tuple[np.ndarray, float]:
    # Custom callback for generation completion
    def on_generation(ga_instance):
        ga_instance.pbar.update(1)
        # Reset and close solutions progress bar
        if hasattr(ga_instance, "pbar_solutions"):
            ga_instance.pbar_solutions.close()
        # Create new solutions progress bar for next generation
        ga_instance.pbar_solutions = tqdm(
            total=num_solutions,
            desc=f"Solutions (gen {ga_instance.generations_completed + 1})",
            leave=False,
        )

    # Create progress bar for generations
    pbar = tqdm(total=num_generations, desc="Generations")

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
        on_generation=on_generation,
    )

    # Attach progress bar to instance so callback can access it
    ga_instance.pbar = pbar
    # Create initial solutions progress bar
    ga_instance.pbar_solutions = tqdm(
        total=num_solutions, desc="Solutions (gen 1)", leave=False
    )

    ga_instance.run()

    # Clean up progress bars
    pbar.close()
    if hasattr(ga_instance, "pbar_solutions"):
        ga_instance.pbar_solutions.close()

    print(f"\nBest fitness achieved: {ga_instance.best_solution()[1]}")

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

    pattern, fitness = runGeneticAlgorithm(100, 500)
    print(f"Best fitness: {fitness}")
    displayGrid(pattern)


if __name__ == "__main__":
    main()
