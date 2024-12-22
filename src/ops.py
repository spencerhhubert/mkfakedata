from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from entropy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from grid import *


def applyOperation(params: GridOpParams, op_call: GridOpCall, record: bool) -> str:
    op = params.grid_ops[op_call.op_idx]
    op.func(params, *op_call.params)
    if record:
        params.grid.history.append(op_call)
    op_name = op.func.__name__
    return op_name


def noOp(params: GridOpParams) -> None:
    pass


def moveRelative(params: GridOpParams, *deltas: float) -> None:
    params.grid.move(list(deltas))


def moveToRandomPlace(params: GridOpParams) -> None:
    new_position = [np.random.randint(0, s) for s in params.grid.grid.shape]
    params.grid.move(new_position)


def propagateFromPoint(params: GridOpParams, strength: float, value: float) -> None:
    indices = np.indices(params.grid.grid.shape)
    squared_diffs = sum(
        (ind - pos) ** 2 for ind, pos in zip(indices, params.grid.position)
    )
    distances = np.sqrt(squared_diffs)
    mask = distances < strength
    params.grid.grid[mask] = value


def line(params: GridOpParams, value: float, *deltas: float) -> None:
    # Convert deltas to target position
    target_pos = []
    for p, d, s in zip(params.grid.position, deltas, params.grid.grid.shape):
        new_p = int(p + d) % s
        target_pos.append(new_p)
    target_pos = tuple(target_pos)

    # Get coordinates as arrays
    start = np.array(params.grid.position)
    end = np.array(target_pos)

    # Calculate number of points needed
    distance = np.max(np.abs(end - start))
    if distance == 0:
        return

    # Create evenly spaced points along line
    t = np.linspace(0, 1, int(distance) + 1)
    points = np.array([start[None,:] * (1-t)[:,None] + end[None,:] * t[:,None]], dtype=int)
    points = points.reshape(-1, len(start))

    # Handle wrapping
    points = points % params.grid.grid.shape

    # Set all points to the specified value
    params.grid.grid[tuple(points.T)] = value


def rectangularFill(params: GridOpParams, value: float, *sizes: float) -> None:
    ndims = len(params.grid.grid.shape)
    sizes = list(sizes)
    while len(sizes) < ndims:
        sizes.append(sizes[-1] if sizes else 0.5)
    ranges = []
    for pos, dim_size, size in zip(params.grid.position, params.grid.grid.shape, sizes):
        actual_size = int(size * dim_size)
        start = max(0, pos - actual_size)
        end = min(dim_size, pos + actual_size + 1)
        ranges.append(slice(start, end))
    params.grid.grid[tuple(ranges)] = value


def remove(params: GridOpParams, radius: float) -> None:
    indices = np.indices(params.grid.grid.shape)
    squared_diffs = sum(
        (ind - pos) ** 2 for ind, pos in zip(indices, params.grid.position)
    )
    distances = np.sqrt(squared_diffs)
    mask = distances < (radius * np.min(params.grid.grid.shape))
    params.grid.grid[mask] = 0


def moduloFill(
    params: GridOpParams, direction: float, step: float, value: float
) -> None:
    ndims = len(params.grid.position)
    dim_idx = int(direction * ndims) % ndims
    actual_step = max(2, int(step * 10))
    for i in range(params.grid.grid.shape[dim_idx]):
        if i % actual_step == 0:
            pos = list(params.grid.position)
            pos[dim_idx] = i
            if all(0 <= p < s for p, s in zip(pos, params.grid.grid.shape)):
                params.grid.grid[tuple(pos)] = value


def rectangle(params: GridOpParams, size: float, value: float) -> None:
    rect_size = int(size * np.min(params.grid.grid.shape) / 2)
    ranges = []
    for pos, dim_size in zip(params.grid.position, params.grid.grid.shape):
        start = max(0, pos - rect_size)
        end = min(dim_size, pos + rect_size + 1)
        ranges.append(slice(start, end))
    params.grid.grid[tuple(ranges)] = value


def singleFill(params: GridOpParams, should_fill: float) -> None:
    value = 1 if should_fill >= 0.5 else 0
    params.grid.grid[params.grid.position] = value


def place(params: GridOpParams, value: float) -> None:
    params.grid.grid[params.grid.position] = value


def repeatLast(params: GridOpParams) -> None:
    if len(params.grid.history) < 2:
        return
    # Temporarily set a shorter history when calling applyOperation
    original_history = params.grid.history
    params.grid.history = params.grid.history[:-1]

    last_op = params.grid.history[-1]
    applyOperation(params, last_op, False)

    # Restore the full history
    params.grid.history = original_history


def repeatLastN(params: GridOpParams, n: float) -> None:
    if len(params.grid.history) < 2:
        return

    if isinstance(n, float):
        num_ops = max(1, min(len(params.grid.history) - 1, int(n * 10)))
    else:
        num_ops = max(1, min(len(params.grid.history) - 1, n))

    # Temporarily set a shorter history
    original_history = params.grid.history
    params.grid.history = params.grid.history[:-1]

    # Get the last N ops from the shortened history
    for i in range(num_ops):
        if i >= len(params.grid.history):
            break
        op = params.grid.history[-(num_ops - i)]
        if params.grid_ops[op.op_idx].func.__name__ == "repeatLastN":
            continue
        applyOperation(params, op, False)

    # Restore the full history
    params.grid.history = original_history


def reflect(params: GridOpParams, axis: float) -> None:
    num_axes = len(params.grid.grid.shape)
    axis_idx = int(axis * num_axes) % num_axes
    params.grid.grid = np.flip(params.grid.grid, axis=axis_idx)


def rotate(params: GridOpParams, k: float, axis1: float, axis2: float) -> None:
    num_axes = len(params.grid.grid.shape)
    if num_axes < 2:
        # Not enough dimensions to perform rotation
        return

    # Get axis indices from axis1 and axis2 parameters
    axis1_idx = int(axis1 * num_axes) % num_axes

    # Create a list of possible axes excluding axis1_idx
    remaining_axes = list(range(num_axes))
    remaining_axes.remove(axis1_idx)

    # Map axis2 to one of the remaining axes
    axis2_idx = int(axis2 * (num_axes - 1)) % (num_axes - 1)
    axis2_idx = remaining_axes[axis2_idx]

    # Ensure rotation steps k is between 1 and 4 (90-degree increments)
    k = int(k * 4) % 4
    if k == 0:
        k = 1  # Rotate at least once

    # Perform the rotation
    params.grid.grid = np.rot90(params.grid.grid, k=k, axes=(axis1_idx, axis2_idx))
