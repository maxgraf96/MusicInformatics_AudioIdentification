from pathlib import Path

import numpy as np

import Database_S
import Fingerprint_S
import os
from Constants import clf_name, query_recordings

# Path to wavs
wav_path = "data" + os.path.sep + query_recordings

def analyse(path):
    fp_q = Fingerprint_S.compute_fingerprint(path)
    results = Database_S.search(fp_q)
    if len(results) == 0:
        return False, False

    # Extract filenames and compare
    path_filename = str(path).split(os.path.sep)[-1].split('-')[0]
    best_result_filename = results[0].split(os.path.sep)[-1][:-4]
    correct = False
    correct_in_three = False
    if path_filename == best_result_filename:
        print("Correct. Yay!")
        correct = True
        correct_in_three = True

    for result in results:
        current_result_filename = result.split(os.path.sep)[-1][:-4]
        if path_filename == current_result_filename:
            correct_in_three = True
    return correct, correct_in_three

def test():
    wavs = Path(wav_path).rglob("*.wav")
    correct = 0
    correct_of_three = 0
    counter = 0
    for wav in wavs:
        is_correct, is_correct_in_three = analyse(str(wav))
        correct += 1 if is_correct else 0
        correct_of_three += 1 if is_correct_in_three else 0
        counter += 1

    print("Correct: " + str(correct) + " / " + str(counter) + " = " + str(round(100 * correct / counter, 2)) + "%")
    print("Correct in three: " + str(correct_of_three) + " / " + str(counter))

Database_S.initialise()

# analyse("data\\query_recordings\\classical.00000-snippet-10-0.wav")
test()

