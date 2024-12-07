from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from entropy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class Grid:
    def __init__(self, shape: Tuple[int, ...]):
        self.grid = np.zeros(shape)
        self.position = tuple(np.random.randint(0, s) for s in shape)
        # self.position = tuple(s // 2 for s in shape)
        # Track both grid changes and position changes
        self.operation_history = []  # List of (positions_changed, values_changed, old_pos, new_pos)

    def record(
        self,
        changed_positions: List[Tuple[int, ...]],
        changed_values: List[float],
        old_position: Tuple[int, ...],
        new_position: Tuple[int, ...],
    ):
        self.operation_history.append(
            (changed_positions, changed_values, old_position, new_position)
        )
        if len(self.operation_history) > 10:
            self.operation_history.pop(0)

    def move(self, deltas: List[float]):
        old_position = self.position
        new_pos = []
        for i, (p, d, s) in enumerate(zip(self.position, deltas, self.grid.shape)):
            # new_p = int(p + d * s)  # Scale movement by dimension size
            new_p = int(p + d)  # dont scale if ints
            new_p = new_p % s  # Wrap around using modulo
            new_pos.append(new_p)
        self.position = tuple(new_pos)
        # Record position change with empty grid changes
        self.record([], [], old_position, self.position)


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
