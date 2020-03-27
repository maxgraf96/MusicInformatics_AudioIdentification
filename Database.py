import os
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import Fingerprint
from Constants import database_recordings, generated, clf_name

# Path to wavs
wav_path = "data" + os.path.sep + database_recordings
# Path to csvs
csv_path = "data" + os.path.sep + generated

def query(fp_query):
    # Load classifier
    clf = load(clf_name)
    predictions = []
    for print in range(fp_query.shape[1]):
        current = fp_query[:, print]
        predictions.append(clf.predict([current]))

    return predictions

def initialise():
    """
    Calculates fingerprints for all recordings in database and stores them in CSVs
    :return:
    """
    # Get all wav paths
    wavs = Path(wav_path).rglob("*.wav")
    # Get number of wavs
    number_of_wavs = len([_ for _ in wavs])  # This call "invalidates" the pathlib object

    # Try to get csv files (only works if they were created previously)
    number_of_csvs = 0
    try:
        csvs = Path(csv_path).rglob("*.csv")
        # Get number of csvs
        number_of_csvs = len([_ for _ in csvs])  # This call "invalidates" the pathlib object

        if number_of_wavs == number_of_csvs:
            print("All fingerprints are already stored in the database...")
            return
    except:
        print("No CSV files found. Generating DB fingerprints anew...")

    if number_of_csvs != number_of_wavs:
        print("Number of CSV files found doesn't match number of wav files. Generating missing fingerprints...")
        # Reload pathlib objects
        wavs = Path(wav_path).rglob("*.wav")
        csvs = Path(csv_path).rglob('*.csv')

        # Number of freshly created fingerprints
        new_fingerprints = 0

        # Generate missing / all fingerprints
        for wav in wavs:
            # Convert path to string
            wav_str = str(wav)
            # Format
            wav_str = wav_str[wav_str.rfind(os.path.sep):-4].replace(os.path.sep, "") + ".csv"

            # Only create if there is no fingerprint for that file
            if not exists_in_database(wav_str):
                fingerprint = Fingerprint.compute_fingerprint(str(wav))
                Fingerprint.save_fingerprint_to_csv(fingerprint, wav_str)
                new_fingerprints += 1

        print("Done creating fingerprints. New: " + str(new_fingerprints))

def train_classifier():
    n_neighbors = 7
    neigh = KNeighborsClassifier(n_neighbors)

    # Create input data: All wav fingerprints
    X = []
    y = []
    # Load csvs
    csvs = Path(csv_path).rglob("*.csv")
    # Add each fingerprint to training data
    for csv in csvs:
        fingerprint = Fingerprint.read_fingerprint_from_csv(csv)
        for column in range(fingerprint.shape[1]):
            current = fingerprint[:, column]
            X.append(current)
            # Store both the csv and the index of the current print as label
            y.append((str(csv), column))

    # Train k nearest neighbors classifier
    neigh.fit(X, y)

    return neigh

def save_classifier(clf):
    dump(clf, clf_name)

def exists_in_database(wav):
    return os.path.exists("data" + os.path.sep + generated + os.path.sep + wav)