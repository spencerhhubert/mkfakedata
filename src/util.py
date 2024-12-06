from pathlib import Path
import json
import numpy as np
from entropy import *
import matplotlib.pyplot as plt


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
    if grid.ndim == 2:
        plt.figure(figsize=(8, 8))

        # Use min/max of data for color scaling
        vmin, vmax = np.min(grid), np.max(grid)

        # If all values are the same, adjust range to prevent division by zero
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        plt.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Value")

        # Add grid lines
        plt.grid(True, which="major", color="black", linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])

        plt.title(f"Grid Values: min={vmin:.2f}, max={vmax:.2f}")
        plt.show()
    else:
        print("Grid shape:", grid.shape)
