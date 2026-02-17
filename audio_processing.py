import librosa
import numpy as np

def extract_energy(video_path):
    audio, sr = librosa.load(video_path, sr=None)
    energy = librosa.feature.rms(y=audio)[0]
    return energy, sr
