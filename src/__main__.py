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
target_entropy = 5
max_gene_value = 100
min_gene_value = 0
gene_type = int
operation_counter = {}


def noOp(params: GridOpParams) -> None:
    pass


GridOps = [
    GridOp(noOp, 0),
    GridOp(moveRelative, len(shape)),
    GridOp(moveToRandomPlace, 0),
    GridOp(place, 1),
    # GridOp(remove, 1),
    GridOp(repeatLast, 0),
    GridOp(repeatLastN, 1),
    GridOp(rectangularFill, len(shape) + 1),
    # GridOp(propagateFromPoint, 2),
    # GridOp(line, 3),
    # GridOp(fill, 2),
    # GridOp(moduloFill, 3),
    # GridOp(rectangle, 2),
    # GridOp(repeatNthAgo, 2),
]


def decodeGenesToOperations(genes: List[float]) -> List[GridOpCall]:
    ops = []
    i = 0
    while i < len(genes):
        if gene_type == int:
            # Direct index selection, use noop (index 0) if out of range
            op_idx = genes[i] if 0 <= genes[i] < len(GridOps) else 0
        else:
            # Original normalization method
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

    op_name = op.func.__name__
    operation_counter[op_name] = operation_counter.get(op_name, 0) + 1


def applyOperationSequence(
    shape: Tuple[int, ...], operations: List[GridOpCall]
) -> np.ndarray:
    grid = Grid(shape)
    for op_call in operations:
        applyOperation(grid, op_call)
    return grid.grid


def calcFitness(ga_instance, solution: list, solution_idx: int) -> float:
    if hasattr(ga_instance, "pbar_solutions"):
        ga_instance.pbar_solutions.update(1)

    operations = decodeGenesToOperations(solution)
    pattern = applyOperationSequence(shape, operations)
    current_entropy = calcShannonEntropy(pattern)
    entropy_score = 1.0 / (1.0 + abs(target_entropy - current_entropy))
    total_score = 1 * entropy_score

    # Initialize logging dict if it doesn't exist
    if not hasattr(ga_instance, "logging"):
        ga_instance.logging = {}

    # Update if this is the best score we've seen
    if not ga_instance.logging or total_score > ga_instance.logging.get(
        "fitness", -float("inf")
    ):
        ga_instance.logging = {
            "pattern": pattern,
            "entropy": current_entropy,
            "fitness": total_score,
        }

    return total_score


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

        fitness_history.append(ga_instance.logging["fitness"])
        entropy_history.append(ga_instance.logging["entropy"])

        ga_instance.pbar.set_description(
            f"Generations (fitness: {fitness_history[-1]:.3f}, entropy: {ga_instance.logging['entropy']:.3f})"
        )

    pbar = tqdm(total=num_generations, desc="Generations")

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=4,
        num_genes=num_genes,
        init_range_low=min_gene_value,
        init_range_high=max_gene_value,
        fitness_func=calcFitness,
        sol_per_pop=num_solutions,
        gene_type=gene_type,
        mutation_percent_genes=20,
        mutation_type="random",
        crossover_type="two_points",
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
    parser.add_argument(
        "--generations", type=int, default=100, help="Number of generations to run"
    )
    parser.add_argument(
        "--genes", type=int, default=200, help="Number of genes per solution"
    )
    args = parser.parse_args()

    generations = args.generations
    genes = args.genes

    pattern, fitness, fitness_history, entropy_history = runGeneticAlgorithm(
        genes, generations
    )
    plotFitness(fitness_history)
    plotEntropy(entropy_history, target_entropy)
    plotOperationUsage(operation_counter)
    displayGrid(pattern)


if __name__ == "__main__":
    main()
