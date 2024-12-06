# alternative to the ops file that is more simple. no float values.

from typing import List, Tuple, Callable
import numpy as np
from grid import *


def binaryPropagateFromPoint(
    params: GridOpParams, strength: float, should_fill: float
) -> None:
    old_grid = params.grid.grid.copy()
    value = 1 if should_fill >= 0.5 else 0

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


def binaryLine(
    params: GridOpParams, direction: float, should_fill: float, length: float
) -> None:
    old_grid = params.grid.grid.copy()
    value = 1 if should_fill >= 0.5 else 0
    ndims = len(params.grid.position)

    dim_idx = int(direction * ndims) % ndims
    step_sign = 1 if direction * ndims % 1 < 0.5 else -1

    max_len = params.grid.grid.shape[dim_idx]
    actual_length = int(length * max_len)

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


def binaryFill(params: GridOpParams, radius: float, should_fill: float) -> None:
    old_grid = params.grid.grid.copy()
    value = 1 if should_fill >= 0.5 else 0

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


def binaryModuloFill(
    params: GridOpParams, direction: float, step: float, should_fill: float
) -> None:
    old_grid = params.grid.grid.copy()
    value = 1 if should_fill >= 0.5 else 0
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


def binaryRectangle(params: GridOpParams, size: float, should_fill: float) -> None:
    old_grid = params.grid.grid.copy()
    value = 1 if should_fill >= 0.5 else 0

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
