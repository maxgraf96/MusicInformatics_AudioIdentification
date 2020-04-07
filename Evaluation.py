import os
import pickle
from pathlib import Path

import Fingerprint
import Main

import numpy as np

database = {}
ids_names = {}

def evaluate_all():
    global database, ids_names
    # Load database from file
    db_file = open("fingerprints" + os.path.sep + "db_S.pkl", "rb")
    ids_names_file = open("fingerprints" + os.path.sep + "ids_names.pkl", "rb")
    database = pickle.load(db_file)
    ids_names = pickle.load(ids_names_file)

    # Path to wavs
    wav_path = "data" + os.path.sep + "query_recordings"

    wavs = Path(wav_path).rglob("*.wav")
    number_of_query_files = sum(1 for x in wavs)

    f_maxs = []
    map = 0
    counter = 0
    for wav in Path(wav_path).rglob("*.wav"):
        print("Evaluating " + str(counter) + " of " + str(number_of_query_files))
        f_max, p_avg = relevance(str(wav).split(os.path.sep)[-1])
        map += p_avg
        f_maxs.append(f_max)
        counter += 1

    f_max_avg = np.mean(f_maxs)
    map /= number_of_query_files

    print("Average F-measure: " + str(f_max_avg) + ", MAP: " + str(map))

def relevance(query_path):
    # fingerprint = Fingerprint.compute_fingerprint(str(query_path))
    best_three, is_correct = Main.analyse(database, ids_names, str(query_path))

    # Get ground truth song ID
    song_name = query_path.split(os.path.sep)[-1].split('-')[0] + ".wav"

    # Get ground truth best ten
    best_ten_gt, is_correct = Main.analyse(database, ids_names, song_name)

    relevant = np.zeros(len(best_three))
    for i in range(len(best_three)):
        if best_three[i] in best_ten_gt:
            relevant[i] = 1

    # # Get all ground truth hashes
    # gt_hashes = []
    # for k, v in database.items():
    #     for (sID, offset) in v:
    #         if sID == song_id:
    #             gt_hashes.append(k)
    # Compute relevance for each item
    # relevant = {}
    # for item in best_ten:
    #     # Assume that item is irrelevant
    #     relevant[item] = 0
    #
    #     # Go through all the hashes in the fingerprint
    #     for (hash, offset) in fingerprint:
    #         digest = hash.digest()
    #         is_included = False
    #         try:
    #             entries = database[digest]
    #             # Check if current song id is included in the current hash
    #             for (song_id, offset) in entries:
    #                 if best_ten_ids[item] == song_id:
    #                     is_included = True
    #                     break
    #
    #             # Last step: Check if the current hash (from the fingerprint) is actually part of the
    #             # ground truth data
    #             if is_included:
    #                 for h in gt_hashes:
    #                     if h == digest:
    #                         relevant[item] = 1
    #         except KeyError:
    #             # In case of an unknown hash continue
    #             continue

    # Calculate precision and recall for each rank

    # scores = np.array([item[1] for item in relevant.items()])

    precision = np.zeros(len(best_three))
    recall = np.zeros(len(best_three))

    for i in range(1, len(best_three) + 1):
        sum = np.sum(relevant[:i])
        precision[i - 1] = (1 / i) * sum
        recall[i - 1] = sum

    # Calculate F-measure
    f_measures = []
    for i in range(len(best_three)):
        if precision[i] == 0 and recall[i] == 0:
            f_measures.append(0)
        else:
            f_measures.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))

    # Calculate maximum f measure
    f_max = np.max(f_measures)

    # Calculate average precision
    p_avg = np.sum(precision * relevant) / 1

    return f_max, p_avg




# Evaluate
# relevance("pop.00050-snippet-10-0.wav")
evaluate_all()