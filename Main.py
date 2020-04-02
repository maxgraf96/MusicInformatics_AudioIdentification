from pathlib import Path

import numpy as np
import Fingerprint
import Database
import os
from Constants import clf_name, query_recordings


def analyse(path):
    fp_query, indices = Fingerprint.compute_fingerprint(path)
    candidates = Database.query(fp_query)

    res = []
    # Get results
    # print("Results:")
    for k, v in candidates.items():
        # Extract name from results path
        name = k[k.rfind(os.path.sep):].replace(os.path.sep, "")[:-4]
        res.append(name)
        print(k, v)

    # Check if correct
    if len(res) > 0 and res[0] in path:
        # Great success
        print("Correct! Yay!")

    print()

def test():
    # Path to wavs
    wav_path = "data" + os.path.sep + query_recordings
    wavs = Path(wav_path).rglob("*.wav")
    counter = 0
    for wav in wavs:
        if counter >= 50:
            break
        analyse(str(wav))
        counter += 1

# Initialise database
Database.initialise()

# Train classifier if it doesn't exist yet
# if not os.path.exists(clf_name):
clf = Database.train_classifier()
Database.save_classifier(clf)

# analyse("data/query_recordings/pop.00050-snippet-10-0.wav")
# analyse("data/database_recordings/pop.00050.wav")
# analyse("data/myown/pop.00050.wav")
test()