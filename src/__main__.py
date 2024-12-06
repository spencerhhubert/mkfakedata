from entropy import *
from grid import *
from ops import *
from binary_ops import *
from util import *
from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from tqdm import tqdm
import time


shape = (10, 10, 10)
target_entropy = 10

GridOps = [
    GridOp(moveRelative, len(shape)),
    GridOp(place, 1),
    GridOp(remove, 1),
    GridOp(repeatLast, 0),
    GridOp(propagateFromPoint, 2),
    # GridOp(line, 3),
    # GridOp(fill, 2),
    # GridOp(moduloFill, 3),
    # GridOp(rectangle, 2),
    # GridOp(repeatNthAgo, 2),
]

# playing with binary ops instead
# GridOps = [
#     GridOp(binaryPropagateFromPoint, 2),
#     GridOp(moveRelative, len(shape)),  # kept original since it's movement
#     GridOp(binaryLine, 3),
#     GridOp(binaryFill, 2),
#     GridOp(remove, 1),  # kept original since it just sets to 0
#     GridOp(binaryModuloFill, 3),
#     GridOp(binaryRectangle, 2),
#     GridOp(repeatNthAgo, 2),  # kept original since it copies existing values
# ]


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


def calcEntropyScore(pattern: np.ndarray, target_entropy: float) -> float:
    current_entropy = calcShannonEntropy(pattern)
    return 1.0 / (1.0 + abs(target_entropy - current_entropy))



def calcFitness(ga_instance, solution: list, solution_idx: int) -> float:
    if hasattr(ga_instance, "pbar_solutions"):
        ga_instance.pbar_solutions.update(1)

    operations = decodeGenesToOperations(solution)
    pattern = applyOperationSequence(shape, operations)

    entropy_score = calcEntropyScore(pattern, target_entropy)

    return entropy_score  # * adjacency_score


def runGeneticAlgorithm(
    num_genes: int = 100, num_generations: int = 200, num_solutions: int = 100
) -> Tuple[np.ndarray, float, List[float], List[float]]:
    fitness_history = []
    entropy_history = []

    def on_generation(ga_instance):
        ga_instance.pbar.update(1)
        if hasattr(ga_instance, "pbar_solutions"):
            ga_instance.pbar_solutions.close()
        ga_instance.pbar_solutions = tqdm(
            total=num_solutions,
            desc=f"Solutions (gen {ga_instance.generations_completed + 1})",
            leave=False,
        )

        best_solution = ga_instance.best_solution()[0]
        best_ops = decodeGenesToOperations(best_solution)
        best_pattern = applyOperationSequence(shape, best_ops)
        current_entropy = calcShannonEntropy(best_pattern)

        fitness_history.append(ga_instance.best_solution()[1])
        entropy_history.append(current_entropy)
        ga_instance.pbar.set_description(
            f"Generations (fitness: {fitness_history[-1]:.3f}, entropy: {current_entropy:.3f})"
        )

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

    ga_instance.pbar = pbar
    ga_instance.pbar_solutions = tqdm(
        total=num_solutions, desc="Solutions (gen 1)", leave=False
    )

    ga_instance.run()

    pbar.close()
    if hasattr(ga_instance, "pbar_solutions"):
        ga_instance.pbar_solutions.close()

    solution, solution_fitness, _ = ga_instance.best_solution()
    operations = decodeGenesToOperations(solution)
    best_pattern = applyOperationSequence(shape, operations)

    return best_pattern, solution_fitness, fitness_history, entropy_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arc", dest="arc_data_path", required=False, help="Path to ARC data"
    )
    args = parser.parse_args()

    pattern, fitness, fitness_history, entropy_history = runGeneticAlgorithm(500, 500)
    print(f"Best fitness: {fitness}")
    plotFitness(fitness_history)
    plotEntropy(entropy_history, target_entropy)
    displayGrid(pattern)


if __name__ == "__main__":
    main()
