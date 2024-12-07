from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from entropy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from grid import *


def applyOperation(params: GridOpParams, op_call: GridOpCall) -> str:
    op = params.grid_ops[op_call.op_idx]
    op.func(params, *op_call.params)
    params.grid.operation_history.append(op_call)
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


def line(params: GridOpParams, direction: float, value: float, length: float) -> None:
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
    if not params.grid.operation_history:
        return
    last_op = params.grid.operation_history[-1]
    applyOperation(params, last_op)


def repeatLastN(params: GridOpParams, n: float) -> None:
    if not params.grid.operation_history:
        return
    num_ops = max(1, min(len(params.grid.operation_history), int(n * 10)))
    for i in range(num_ops):
        if i >= len(params.grid.operation_history):
            break
        op = params.grid.operation_history[-(num_ops - i)]
        applyOperation(params, op)
