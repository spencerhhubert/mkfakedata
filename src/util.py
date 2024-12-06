from pathlib import Path
import json
import numpy as np
from entropy import *


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
