import numpy as np
from scipy.stats import entropy


def calcShannonEntropy(arr):
    values, counts = np.unique(arr, return_counts=True)
    probabilities = counts / len(arr.flatten())
    return -np.sum(probabilities * np.log2(probabilities))


def calcSampleEntropy(arr, pattern_length=2):
    flattened = arr.flatten()
    n = len(flattened)
    patterns = np.array(
        [flattened[i : i + pattern_length] for i in range(n - pattern_length + 1)]
    )
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    return entropy(counts)


def binaryShannonEntropy(array: np.ndarray) -> float:
    # Convert to binary (0 and 1)
    binary = (array != 0).astype(int)

    # Count occurrences
    unique, counts = np.unique(binary, return_counts=True)
    probs = counts / len(binary.flatten())

    # Calculate entropy (only for non-zero probabilities)
    entropy = -np.sum([p * np.log(p) for p in probs if p > 0])
    return entropy


def calcApproxEntropy(arr, m=2, r=0.2):
    flattened = arr.flatten()
    std = np.std(flattened)
    r = r * std
    n = len(flattened)

    def phi(m):
        patterns = np.array([flattened[i : i + m] for i in range(n - m + 1)])
        distances = np.abs(patterns[:, None] - patterns)
        counts = np.sum(np.all(distances <= r, axis=2), axis=1)
        return np.mean(np.log(counts / (n - m + 1)))

    return phi(m) - phi(m + 1)
