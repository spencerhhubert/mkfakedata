from typing import List, Tuple, Callable
import numpy as np
import pygad
import argparse
from entropy import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class GridOp:
    def __init__(self, func: Callable[..., None], param_count: int):
        self.func = func
        self.param_count = param_count


class GridOpCall:
    def __init__(self, op_idx: int, params: List[float]):
        self.op_idx = op_idx
        self.params = params


class Grid:
    def __init__(self, shape: Tuple[int, ...]):
        self.grid = np.zeros(shape)
        self.position = tuple(np.random.randint(0, s) for s in shape)
        # self.position = tuple(s // 2 for s in shape)
        # Track both grid changes and position changes
        self.history = []  # List of (positions_changed, values_changed, old_pos, new_pos)

    def move(self, deltas: List[float]):
        self.position = tuple(
            int(p + d) % s for p, d, s in zip(self.position, deltas, self.grid.shape)
        )


class GridOpParams:
    def __init__(self, grid: Grid, grid_ops: List[GridOp]):
        self.grid = grid
        self.grid_ops = grid_ops
