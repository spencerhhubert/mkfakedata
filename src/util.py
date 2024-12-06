from pathlib import Path
import json
from typing import List
import numpy as np
from entropy import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def loadArcData(data_path):
    json_files = list(Path(data_path).rglob("*.json"))
    all_data = []

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        for split in ["train", "test"]:
            for item in data[split]:
                item["input"] = np.array(item["input"])
                item["output"] = np.array(item["output"])

        all_data.append(data)

    return all_data


def writeArcEntropyCalculations(arc_data):
    entropy_data = []
    for data in arc_data:
        file_entropy = {"train": [], "test": []}

        for split in ["train", "test"]:
            for item in data[split]:
                item_entropy = {
                    "shannon": calcShannonEntropy(item["input"]),
                    "sample": calcSampleEntropy(item["input"]),
                    "approx": calcApproxEntropy(item["input"]),
                }
                file_entropy[split].append(item_entropy)

        entropy_data.append(file_entropy)

    with open("arc-entropy.json", "w") as f:
        json.dump(entropy_data, f, indent=2)


def displayGrid(grid: np.ndarray) -> None:
    if grid.ndim == 1:
        plt.figure(figsize=(8, 4))
        plt.plot(grid)
        plt.grid(True)
        plt.show()
    elif grid.ndim == 2:
        plt.figure(figsize=(8, 8))
        vmin, vmax = np.min(grid), np.max(grid)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        plt.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Value")
        plt.grid(True, which="major", color="black", linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
        plt.title(f"Grid Values: min={vmin:.2f}, max={vmax:.2f}")
        plt.show()
    elif grid.ndim == 3:
        x, y, z = np.indices(grid.shape)
        values = grid.flatten()
        normalized_values = (values - np.min(values)) / (
            np.max(values) - np.min(values)
        )

        fig = go.Figure(
            data=go.Scatter3d(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                mode="markers",
                marker=dict(size=50, color=values, colorscale="viridis", opacity=0.8),
            )
        )
        fig.show()
    else:
        print("Grid shape:", grid.shape)


def plotFitness(fitness_history: List[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.title("Fitness Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()


def plotEntropy(entropy_history: List[float], target_entropy) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(entropy_history)
    plt.axhline(y=target_entropy, color="r", linestyle="--", label="Target")
    plt.title("Entropy Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Entropy")
    plt.legend()
    plt.show()


def plotOperationUsage(operation_counter):
    plt.figure(figsize=(12, 6))
    names = list(operation_counter.keys())
    counts = list(operation_counter.values())

    plt.bar(names, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Operation Usage Count")
    plt.ylabel("Times Used")
    plt.tight_layout()
    plt.show()
