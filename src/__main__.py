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
target_entropy = 0.5
target_density = 10
target_components = 100
gene_type = int
operation_counter = {}


GridOps = [
    GridOp(noOp, 0),
    GridOp(moveRelative, len(shape)),
    # GridOp(moveToRandomPlace, 0),
    GridOp(place, 1),
    # GridOp(remove, 1),
    # GridOp(repeatLast, 0),
    GridOp(repeatLastN, 1),
    # GridOp(rectangularFill, len(shape) + 1),
    # GridOp(propagateFromPoint, 2),
    # GridOp(line, len(shape) + 1),
    # GridOp(fill, 2),
    # GridOp(moduloFill, 3),
    # GridOp(rectangle, 2),
    # GridOp(repeatNthAgo, 2),
    # GridOp(reflect, 1)
    # GridOp(rotate, 3)
]

max_params = max(op.param_count for op in GridOps)

max_gene_value = 10
min_gene_value = 0


def decodeGenesToOperations(genes: List[float]) -> List[GridOpCall]:
    ops = []
    i = 0
    while i + max_params < len(genes):
        if gene_type == int:
            op_idx = genes[i] if 0 <= genes[i] < len(GridOps) else 0
        else:
            op_idx = int(genes[i] * len(GridOps))
            op_idx = max(0, min(op_idx, len(GridOps) - 1))

        # Only take the number of parameters this operation needs
        needed_params = GridOps[op_idx].param_count
        params = genes[i + 1:i + 1 + needed_params]
        ops.append(GridOpCall(op_idx, params))
        i += 1 + max_params

    return ops


def printGenes(solution: list) -> None:
    print("\nOptimal Solution Operations:")
    operations = decodeGenesToOperations(solution)
    for op_call in operations:
        op = GridOps[op_call.op_idx]
        print(f"{op.func.__name__}({', '.join(map(str, op_call.params))})")


def applyOperationSequence(
    shape: Tuple[int, ...], operations: List[GridOpCall]
) -> np.ndarray:
    params = GridOpParams(Grid(shape), GridOps)
    for op_call in operations:
        op_name = applyOperation(params, op_call, True)
        operation_counter[op_name] = operation_counter.get(op_name, 0) + 1
    return params.grid.grid


def calcFitness(ga_instance, solution: list, solution_idx: int) -> float:
    try:
        # Generate pattern from solution
        operations = decodeGenesToOperations(solution)
        pattern = applyOperationSequence(shape, operations)

        # Entropy calculation using target_entropy
        shannon_entropy = calcShannonEntropy(pattern)
        entropy_diff = abs(target_entropy - shannon_entropy)
        entropy_score = 1 / (1 + entropy_diff)

        # Density calculation
        non_zero_count = np.count_nonzero(pattern)
        density = non_zero_count / pattern.size
        density_diff = abs(target_density - density)
        density_score = 1 / (1 + density_diff)

        # Count connected components (works for any dimension)
        num_components = countConnectedComponents(pattern)
        components_diff = abs(target_components - num_components)
        components_score = 1 / (1 + components_diff)

        # Diversity score (number of unique values)
        num_unique_values = len(np.unique(pattern))
        max_possible_values = max(pattern.size, 1)  # Avoid division by zero
        diversity_score = num_unique_values / max_possible_values

        # Weights for different metrics
        entropy_weight = 0.5
        density_weight = 0.2
        components_weight = 0.2
        diversity_weight = 0.1

        # Combine scores with weights
        fitness = (
            entropy_weight * entropy_score +
            density_weight * density_score +
            components_weight * components_score +
            diversity_weight * diversity_score
        )

        # Total score without operation penalty
        total_score = fitness

        # Optionally, update logging
        if not hasattr(ga_instance, "logging") or total_score > ga_instance.logging.get("fitness", -float("inf")):
            ga_instance.logging = {
                "pattern": pattern,
                "entropy": shannon_entropy,
                "fitness": total_score,
            }

        return total_score
    except Exception as e:
        print(f"Error in fitness function for solution {solution_idx}: {e}")
        return -np.inf



def runGeneticAlgorithm(
    num_genes: int, num_generations: int, num_solutions: int, parents_mating: int
) -> Tuple[List[np.ndarray], float, List[float], List[float]]:
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
        num_parents_mating=parents_mating,
        num_genes=num_genes,
        init_range_low=min_gene_value,
        init_range_high=max_gene_value,
        fitness_func=calcFitness,
        sol_per_pop=num_solutions,
        gene_type=gene_type,
        parent_selection_type="tournament",
        mutation_probability=0.01,
        mutation_percent_genes=10,
        mutation_type="random",
        crossover_type="single_point",
        crossover_probability=0.8,
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
    # printGenes(solution)
    operations = decodeGenesToOperations(solution)
    patterns = [applyOperationSequence(shape, operations) for _ in range(10)]

    return patterns, solution_fitness, fitness_history, entropy_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arc", dest="arc_data_path", required=False, help="Path to ARC data"
    )
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument(
        "--genes",
        type=int,
        default=2000,
    )
    parser.add_argument("--solutions", type=int, default=50)
    parser.add_argument("--parents", type=int, default=4)
    global target_entropy
    parser.add_argument("--entropy", type=float, default=target_entropy)
    args = parser.parse_args()

    generations = args.generations
    genes = args.genes
    solutions = args.solutions
    parents = args.parents
    target_entropy = args.entropy

    patterns, fitness, fitness_history, entropy_history = runGeneticAlgorithm(
        genes, generations, solutions, parents
    )
    plotFitness(fitness_history)
    # plotEntropy(entropy_history, target_entropy)
    # plotOperationUsage(operation_counter)
    displayGrids(patterns)


if __name__ == "__main__":
    main()
