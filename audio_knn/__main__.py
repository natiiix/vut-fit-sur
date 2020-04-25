from os import listdir, path
import pickle

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.spatial.distance import euclidean as euclid_dist

MODULE_DIR = path.dirname(path.abspath(__file__))
PICKLE_PATH = path.join(MODULE_DIR, 'known_wavs.pickle')
DATA_DIR = path.join(MODULE_DIR, 'data')

LABEL_TARGET = 1
LABEL_NON_TARGET = 0

KNN_K = 3


class WavFile:
    def __init__(self, dirPath, fileName, category):
        self.dir = dirPath
        self.name = fileName
        self.path = path.join(self.dir, self.name)
        self.category = category

        fs, data = wavfile.read(self.path)

        _, _, sxx = spectrogram(data / (2 ** 15), fs, nperseg=400, noverlap=240, nfft=511)
        self.features = np.mean(sxx, axis=1)


def read_wav_files(subdir_name, category):
    subdir_path = path.join(DATA_DIR, subdir_name)
    return [WavFile(subdir_path, f, category) for f in listdir(subdir_path) if f.endswith(".wav")]


def predict(known, unknown):
    predictions = [y.category for y in sorted(known, key=lambda x: euclid_dist(unknown.features, x.features))[:KNN_K]]
    return max([(k, predictions.count(k) / len(predictions)) for k in set(predictions)], key=lambda pair: pair[1])


if __name__ == "__main__":
    if path.isfile(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            known_wavs = pickle.load(f)

    else:
        known_wavs = []
        known_wavs.extend(read_wav_files('target_train', LABEL_TARGET))
        known_wavs.extend(read_wav_files('non_target_train', LABEL_NON_TARGET))
        known_wavs.extend(read_wav_files('target_dev', LABEL_TARGET))
        known_wavs.extend(read_wav_files('non_target_dev', LABEL_NON_TARGET))

        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(known_wavs, f)

    for w in read_wav_files('eval', None):
        pred, confidence = predict(known_wavs, w)
        print(w.name.rsplit('.', 1)[0], round(confidence if pred is LABEL_TARGET else (1 - confidence), 2), pred)
