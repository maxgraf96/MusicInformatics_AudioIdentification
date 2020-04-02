import os
import sys

import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from joblib import dump, load
import Fingerprint
from Constants import database_recordings, generated, clf_name

# Path to wavs
wav_path = "data" + os.path.sep + database_recordings
# Path to npys
npy_path = "data" + os.path.sep + generated

labels = []

def query(fp_query):
    # Load classifier
    clf = load(clf_name)
    predictions = []
    for fp in range(len(fp_query)):
        current = fp_query[fp]
        current = current.flatten()
        # Get k nearest neighbors
        knn = clf.kneighbors([current])
        current_predictions = []
        for neighbor in knn[1][0]:
            label = labels[neighbor]
            current_predictions.append(label)
        # Get data
        predictions.append(current_predictions)

    # Extract elements
    candidates = {}
    for prediction in predictions:
        # Every prediction here contains a list of candidates (different items) => group them per item
        for piece in prediction:
            name = piece[0]
            time = piece[1]
            if name not in candidates:
                candidates[name] = [time]
            else:
                candidates[name].append(time)

    # Order candidate dict by name for better overview
    candidates = {k: v for k, v in sorted(candidates.items(), key=lambda item: item[0])}
    # candidates = {k: np.sum(np.diff(v)) for k, v in candidates.items()}
    # candidates = {k: v for k, v in candidates.items() if v > 0}

    # Sort by occurence
    candidates = {k: v for k, v in sorted(candidates.items(), reverse=True, key=lambda item: len(item[1]))}
    # Return first three candidates
    return {k: v for k, v in [x for x in candidates.items()][:3]}


def train_classifier():
    n_neighbors = 7
    neigh = KNeighborsClassifier(n_neighbors, p=2, weights='distance', leaf_size=400)

    # Create input data: All wav fingerprints
    X = []
    # Load npys
    npys = Path(npy_path).rglob("*.npy")
    # Add each fingerprint to training data
    for npy in npys:
        fingerprint = Fingerprint.read_fingerprint(npy)
        for column in range(fingerprint.shape[0]):
            current = fingerprint[column]
            current = current.flatten()
            X.append(current)
            # Store both the npy and the index of the current print as label
            labels.append((str(npy), column))

    # Shuffle data before fitting
    # X, y = shuffle(X, labels)

    # Train k nearest neighbors classifier
    neigh.fit(X, labels)

    return neigh


def save_classifier(clf):
    dump(clf, clf_name)


def exists_in_database(wav):
    return os.path.exists("data" + os.path.sep + generated + os.path.sep + wav + ".npy")


def initialise():
    """
    Calculates fingerprints for all recordings in database and stores them in NPYs
    :return:
    """
    # Get all wav paths
    wavs = Path(wav_path).rglob("*.wav")
    # Get number of wavs
    number_of_wavs = len([_ for _ in wavs])  # This call "invalidates" the pathlib object

    # Try to get npy files (only works if they were created previously)
    number_of_npys = 0
    try:
        npys = Path(npy_path).rglob("*.npy")
        # Get number of npys
        number_of_npys = len([_ for _ in npys])  # This call "invalidates" the pathlib object

        if number_of_wavs == number_of_npys:
            print("All fingerprints are already stored in the database...")
            return
    except:
        print("No NPY files found. Generating DB fingerprints anew...")

    if number_of_npys != number_of_wavs:
        print("Number of NPY files found doesn't match number of wav files. Generating missing fingerprints...")
        # Reload pathlib objects
        wavs = Path(wav_path).rglob("*.wav")
        npys = Path(npy_path).rglob('*.npy')

        # Number of freshly created fingerprints
        new_fingerprints = 0

        # Generate missing / all fingerprints
        for wav in wavs:
            # Convert path to string
            wav_str = str(wav)
            # Format
            wav_str = wav_str[wav_str.rfind(os.path.sep):-4].replace(os.path.sep, "")

            # Only create if there is no fingerprint for that file
            if not exists_in_database(wav_str):
                fingerprint, indices = Fingerprint.compute_fingerprint(str(wav))
                Fingerprint.save_fingerprint(fingerprint, wav_str)
                new_fingerprints += 1

        print("Done creating fingerprints. New: " + str(new_fingerprints))
