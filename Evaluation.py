import os
import pickle
from pathlib import Path

import numpy as np

import Main

database = {}
ids_names = {}

def evaluate_all():
    """
    Evaluates all query files and prints info
    :return: None
    """
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
    all_f_measures = []
    all_precisions = []
    all_recalls = []
    map = 0
    counter = 0
    for wav in Path(wav_path).rglob("*.wav"):
        print("Evaluating " + str(counter) + " of " + str(number_of_query_files))
        precision, recall, f_measures, f_max, p_avg = evaluate(str(wav).split(os.path.sep)[-1])

        all_precisions.append(precision)
        all_recalls.append(recall)
        f_maxs.append(f_max)
        all_f_measures.append(f_measures)
        map += p_avg
        counter += 1

    avg_all_f = np.mean(all_f_measures, axis=0)
    f_max_avg = np.mean(f_maxs)
    map /= number_of_query_files

    print("Average precision for first three ranks : " + np.array2string(np.mean(all_precisions, axis=0), separator=", "))
    print("Average recall for first three ranks : " + np.array2string(np.mean(all_recalls, axis=0), separator=", "))
    print("Average F-measue values for the first three ranks: " + np.array2string(avg_all_f, separator=", "))
    print("Average maximum F-measure: " + str(f_max_avg))
    print("Mean Average Precision: " + str(map))

def evaluate(query_path):
    """
    Evaluate a given *.wav file
    :param query_path: The path to the *.wav file
    :return: Precision, recall and f_measures for the first three ranks, the maximum f-measure and the average precision
    """
    # Get matching results
    best_three, is_correct = Main.analyse(database, ids_names, str(query_path))

    # Get ground truth song ID
    song_name = query_path.split(os.path.sep)[-1].split('-')[0] + ".wav"

    # Get ground truth best ten
    best_ten_gt, is_correct = Main.analyse(database, ids_names, song_name)

    relevant = np.zeros(len(best_three))
    for i in range(len(best_three)):
        if best_three[i] in best_ten_gt:
            relevant[i] = 1

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

    return precision, recall, f_measures, f_max, p_avg

# Evaluate
# relevance("pop.00050-snippet-10-0.wav")
# evaluate_all()