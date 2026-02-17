import numpy as np

def detect_highlights(energy, percentile=85):
    threshold = np.percentile(energy, percentile)
    return np.where(energy > threshold)[0]
