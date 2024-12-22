import numpy as np
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import label


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


def calcMultiscaleEntropy(arr, scales=[1, 2, 4]):
    total_entropy = 0
    for scale in scales:
        window_sizes = [scale] * arr.ndim
        entropy_list = []
        slices = [range(0, s, ws) for s, ws in zip(arr.shape, window_sizes)]
        for idx in np.ndindex(*[len(s) for s in slices]):
            window_slices = tuple(
                slice(slices[dim][idx[dim]], slices[dim][idx[dim]] + window_sizes[dim])
                for dim in range(arr.ndim)
            )
            window = arr[window_slices]
            if window.size > 0:
                entropy_value = calcShannonEntropy(window)
                entropy_list.append(entropy_value)
        if entropy_list:
            total_entropy += np.mean(entropy_list)
    return total_entropy / len(scales)


# really only works for 2D arrays
def calcGlcmEntropy(arr):
    arr = arr.astype(int)
    max_value = arr.max()
    glcm = graycomatrix(arr, distances=[1], angles=[0], levels=max_value+1, symmetric=True, normed=True)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    return entropy


def calcAutocorrelation(arr):
    arr_flat = arr.flatten()
    arr_mean = np.mean(arr_flat)
    arr_var = np.var(arr_flat)
    autocorr = np.correlate(arr_flat - arr_mean, arr_flat - arr_mean, mode='full')
    autocorr_norm = autocorr / (arr_var * len(arr_flat))
    return autocorr_norm[autocorr_norm.size // 2 + 1]  # First positive lag


def countConnectedComponents(pattern):
    # Define connectivity for N-dimensional array
    structure = np.ones(tuple([3] * pattern.ndim), dtype=int)

    # Label connected components
    labeled, num_features = label(pattern > 0, structure=structure)
    return num_features
