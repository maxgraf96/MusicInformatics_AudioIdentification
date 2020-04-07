import pickle
from pathlib import Path

import numpy as np

import Database
import Fingerprint
import os
from Constants import clf_name, query_recordings

def fingerprint_builder(path_to_db, path_to_fingerprints):
    """
    Creates fingerprints of all files specified in path_to_db
    :param path_to_db: The path to the database files
    :param path_to_fingerprints: The folder where fingerprints will be generated. If this folder does not yet exist
    it will be auto-generated
    :return: None
    """
    # Check if path to db is valid
    if not os.path.exists(path_to_db):
        print("Path to db files does not exist / is invalid. Aborting...")
        return

    # Create folder for saved spectrograms if not exists
    if not os.path.exists(path_to_fingerprints):
        Path(path_to_fingerprints).mkdir(parents=True, exist_ok=True)

    # Don't generate fingerprint database if it already exists
    if os.path.exists(path_to_fingerprints + os.path.sep + 'db_S.pkl') \
            and os.path.exists(path_to_fingerprints + os.path.sep + 'ids_names.pkl'):
        print("Fingerprint database was already generated. It is located in " + path_to_fingerprints)
        print("Aborting database fingerprint generation")
        print("If you want to re-generate the database on purpose, delete the 'db_S.pkl' and 'ids_names.pkl' files.")
        return

    # Initialise database and id to name map objects
    database = {}
    ids_names = {}

    # Get all wav files in db folder
    wavs = Path(path_to_db).rglob("*.wav")

    # Get number of files in folder (this invalidates the pathlib object)
    number_of_db_files = sum(1 for x in wavs)

    # Reset pathlib object
    wavs = Path(path_to_db).rglob("*.wav")

    counter = 0
    for wav in wavs:
        print("Generating fingerprint " + str(counter) + " / " + str(number_of_db_files))
        song_id = counter
        hashes = Fingerprint.compute_fingerprint(wav)
        for h, offset in hashes:
            digest = h.digest()
            if digest in database:
                database[digest].append((song_id, offset))
            else:
                database[digest] = [(song_id, offset)]

        # Associate id to name
        ids_names[song_id] = str(wav)
        counter += 1

    with open(path_to_fingerprints + os.path.sep + 'db_S.pkl', 'wb') as f:
        pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)
        print("Database saved to file 'db_S.pkl' in folder " + str(path_to_fingerprints))

        # Save name ids dict
    with open(path_to_fingerprints + os.path.sep + 'ids_names.pkl', 'wb') as f:
        pickle.dump(ids_names, f, pickle.HIGHEST_PROTOCOL)
        print("Song ID to song name map 'ids_names.pkl' saved in folder " + str(path_to_fingerprints))

    print("Done generating fingerprints")

def audio_identification_(path_to_queryset, path_to_fingerprints, path_to_output_txt):
    if not os.path.exists(path_to_fingerprints) or not os.path.exists(path_to_fingerprints + os.path.sep + "db_S.pkl"):
        print("Path to fingerprint database does not exist / is invalid. Aborting...")
        return

    # Load database from file
    db_file = open(path_to_fingerprints + os.path.sep + "db_S.pkl", "rb")
    ids_names_file = open(path_to_fingerprints + os.path.sep + "ids_names.pkl", "rb")
    database = pickle.load(db_file)
    ids_names = pickle.load(ids_names_file)

    # Clear txt
    txt_file = open(path_to_output_txt, 'w+')

    # Get all wavs in query folder
    query_wavs = Path(path_to_queryset).rglob("*.wav")

    # Get number of files in folder (this invalidates the pathlib object)
    number_of_query_files = sum(1 for x in query_wavs)

    # Reload the wavs (because object is invalidated after counting)
    query_wavs = Path(path_to_queryset).rglob("*.wav")

    wrong_files = []
    correct = 0
    counter = 0
    for wav in query_wavs:
        best_three, is_correct = analyse(database, ids_names, str(wav))
        correct += 1 if is_correct else 0
        if not is_correct:
            wrong_files.append(str(wav))

        # Write query filename and three resulting filenames to txt
        query_filename = str(wav).split(os.path.sep)[-1]
        txt_file.write(query_filename + '\t')
        for result in best_three:
            filename = result.split(os.path.sep)[-1]
            txt_file.write(filename + '\t')

        # Newline
        txt_file.write("\n")

        counter += 1

    print("Done processing. Correct: " + str(round(100 * correct / number_of_query_files, 2)) + "%.")
    print("Wrong files: " + ", ".join(wrong_files))

def analyse(database, ids_names, path):
    fp_q = Fingerprint.compute_fingerprint(path)
    results = Database.search(database, ids_names, fp_q)
    if len(results) == 0:
        return [], False, False

    # Extract filenames and compare
    path_filename = str(path).split(os.path.sep)[-1].split('-')[0]
    best_result_filename = results[0].split(os.path.sep)[-1][:-4]
    correct = False
    if path_filename == best_result_filename:
        print("Correct. Yay!")
        correct = True

    return results, correct

# fingerprint_builder("data/database_recordings", "fingerprints")
# audio_identification_("data/query_recordings", "fingerprints", "output.txt")