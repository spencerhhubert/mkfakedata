from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from entropy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from grid import *


def moveRelative(params: GridOpParams, *deltas: float) -> None:
    params.grid.movePosition(list(deltas))


def propagateFromPoint(params: GridOpParams, strength: float, value: float) -> None:
    old_grid = params.grid.grid.copy()

    indices = np.indices(params.grid.grid.shape)
    squared_diffs = sum(
        (ind - pos) ** 2 for ind, pos in zip(indices, params.grid.position)
    )
    distances = np.sqrt(squared_diffs)
    mask = distances < strength
    params.grid.grid[mask] = value

    changed_positions = list(zip(*np.where(old_grid != params.grid.grid)))
    changed_values = [params.grid.grid[pos] for pos in changed_positions]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def line(params: GridOpParams, direction: float, value: float, length: float) -> None:
    old_grid = params.grid.grid.copy()
    ndims = len(params.grid.position)

    # Ensure dim_idx is within bounds
    dim_idx = int(direction * ndims) % ndims  # This ensures it's always valid
    step_sign = 1 if direction * ndims % 1 < 0.5 else -1

    max_len = params.grid.grid.shape[dim_idx]
    actual_length = int(length * max_len)

    # should this punish the function for going out of the bounds? should it wrap around?
    for i in range(actual_length):
        pos = list(params.grid.position)
        pos[dim_idx] = (pos[dim_idx] + i * step_sign) % params.grid.grid.shape[dim_idx]
        if all(0 <= p < s for p, s in zip(pos, params.grid.grid.shape)):
            params.grid.grid[tuple(pos)] = value

    changed_positions = list(zip(*np.where(old_grid != params.grid.grid)))
    changed_values = [params.grid.grid[pos] for pos in changed_positions]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def fill(params: GridOpParams, radius: float, value: float) -> None:
    old_grid = params.grid.grid.copy()

    indices = np.indices(params.grid.grid.shape)
    squared_diffs = sum(
        (ind - pos) ** 2 for ind, pos in zip(indices, params.grid.position)
    )
    distances = np.sqrt(squared_diffs)
    mask = distances < (radius * np.min(params.grid.grid.shape))
    params.grid.grid[mask] = value

    changed_positions = list(zip(*np.where(old_grid != params.grid.grid)))
    changed_values = [params.grid.grid[pos] for pos in changed_positions]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def remove(params: GridOpParams, radius: float) -> None:
    old_grid = params.grid.grid.copy()

    indices = np.indices(params.grid.grid.shape)
    squared_diffs = sum(
        (ind - pos) ** 2 for ind, pos in zip(indices, params.grid.position)
    )
    distances = np.sqrt(squared_diffs)
    mask = distances < (radius * np.min(params.grid.grid.shape))
    params.grid.grid[mask] = 0

    changed_positions = list(zip(*np.where(old_grid != params.grid.grid)))
    changed_values = [params.grid.grid[pos] for pos in changed_positions]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def moduloFill(
    params: GridOpParams, direction: float, step: float, value: float
) -> None:
    old_grid = params.grid.grid.copy()
    ndims = len(params.grid.position)

    dim_idx = int(direction * ndims) % ndims
    actual_step = max(2, int(step * 10))

    for i in range(params.grid.grid.shape[dim_idx]):
        if i % actual_step == 0:
            pos = list(params.grid.position)
            pos[dim_idx] = i
            if all(0 <= p < s for p, s in zip(pos, params.grid.grid.shape)):
                params.grid.grid[tuple(pos)] = value

    changed_positions = list(zip(*np.where(old_grid != params.grid.grid)))
    changed_values = [params.grid.grid[pos] for pos in changed_positions]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def rectangle(params: GridOpParams, size: float, value: float) -> None:
    old_grid = params.grid.grid.copy()

    rect_size = int(size * np.min(params.grid.grid.shape) / 2)

    ranges = []
    for pos, dim_size in zip(params.grid.position, params.grid.grid.shape):
        start = max(0, pos - rect_size)
        end = min(dim_size, pos + rect_size + 1)
        ranges.append(slice(start, end))

    params.grid.grid[tuple(ranges)] = value

    changed_positions = list(zip(*np.where(old_grid != params.grid.grid)))
    changed_values = [params.grid.grid[pos] for pos in changed_positions]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def repeatNthAgo(params: GridOpParams, n_ago: float, scale: float) -> None:
    if not params.grid.operation_history:  # If history is empty
        return

    n = int(n_ago * len(params.grid.operation_history))
    # Ensure n is valid (between 0 and len-1)
    # should punish?
    n = min(max(0, n), len(params.grid.operation_history) - 1)

    positions, values, old_pos, new_pos = params.grid.operation_history[-n - 1]
    current_pos = np.array(params.grid.position)

    # If it was a move operation
    if not positions and not values:
        # Calculate and apply relative movement
        relative_move = np.array(new_pos) - np.array(old_pos)
        new_position = current_pos + (relative_move * scale)
        params.grid.movePosition([d * scale for d in relative_move])
        return

    # Otherwise handle grid changes
    for pos, val in zip(positions, values):
        old_center = np.mean(positions, axis=0)
        offset = np.array(pos) - old_center
        new_pos = tuple(int(p) for p in current_pos + (offset * scale))

        if all(0 <= p < s for p, s in zip(new_pos, params.grid.grid.shape)):
            params.grid.grid[new_pos] = val


def singleFill(params: GridOpParams, should_fill: float) -> None:
    old_grid = params.grid.grid.copy()
    value = 1 if should_fill >= 0.5 else 0

    params.grid.grid[params.grid.position] = value

    changed_positions = [params.grid.position]
    changed_values = [value]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def place(params: GridOpParams, value: float) -> None:
    old_grid = params.grid.grid.copy()
    params.grid.grid[params.grid.position] = value

    changed_positions = [params.grid.position]
    changed_values = [value]
    params.grid.recordChange(
        changed_positions, changed_values, params.grid.position, params.grid.position
    )


def repeatLast(params: GridOpParams) -> None:
    if not params.grid.operation_history:  # If history is empty
        return

    positions, values, old_pos, new_pos = params.grid.operation_history[-1]
    current_pos = np.array(params.grid.position)

    # If it was a move operation
    if not positions and not values:
        # Calculate and apply relative movement
        relative_move = np.array(new_pos) - np.array(old_pos)
        params.grid.movePosition(relative_move.tolist())
        return

    # Otherwise handle grid changes
    for pos, val in zip(positions, values):
        old_center = np.mean(positions, axis=0)
        offset = np.array(pos) - old_center
        new_pos = tuple(int(p) for p in current_pos + offset)

        if all(0 <= p < s for p, s in zip(new_pos, params.grid.grid.shape)):
            params.grid.grid[new_pos] = val
