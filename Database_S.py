import os
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import Fingerprint_S
from Constants import database_recordings, wav_path

database = {}
ids_names = {}

def search(fp_query):
    matches = {}
    for (hash_q, offset_q) in fp_query:
        hash_q_digest = hash_q.digest()
        try:
            db_lookup = database[hash_q_digest]

            song_id = db_lookup[0]
            db_offset = db_lookup[1]
            if song_id in matches:
                matches[song_id].append((db_offset, offset_q))
            else:
                matches[song_id] = [(db_offset, offset_q)]
        except KeyError:
            # Hash not found in db
            continue

    def takeFirst(elem):
        return elem["song"]

    # Order matches by song id
    matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[0])}

    candidates = {}

    # Group each song into histogram bins
    for song_id in matches.keys():
        pairs = matches[song_id]
        offset_diffs = []
        # Calculate offset diff for each match of current song
        for pair in pairs:
            offset_diffs.append(pair[0] - pair[1])

        most_common = Counter(offset_diffs).most_common(1)
        maximum = most_common[0][1]
        candidates[song_id] = maximum

    # Select best 3
    matches = [k for k, v in sorted(candidates.items(), key=lambda item: item[1], reverse=True)]
    best_3 = matches[:3]

    # Lookup the song names from the ids
    song_names = [ids_names[best] for best in best_3]

    return song_names

def initialise():
    global database
    global ids_names
    # Create full db if not exists
    if not os.path.exists("db_S.pkl"):
        # Get all wav paths

        # Simply use integer as song id for now
        counter = 0
        for wav in Path(wav_path).rglob("*.wav"):
            print("Generating " + str(counter) + " / " + str(300))
            song_id = counter
            hashes = Fingerprint_S.compute_fingerprint(wav)
            for h, offset in hashes:
                database[h.digest()] = (song_id, offset)

            # Associate id to name
            ids_names[song_id] = str(wav)
            counter += 1

        with open('db_S.pkl', 'wb') as f:
            pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)
            print("Database saved to file 'db_S.pkl'")

        # Save name ids dict
        with open('ids_names.pkl', 'wb') as f:
            pickle.dump(ids_names, f, pickle.HIGHEST_PROTOCOL)

    else:
        # Load database from file
        db_file = open("db_S.pkl", "rb")
        ids_names_file = open("ids_names.pkl", "rb")
        database = pickle.load(db_file)
        ids_names = pickle.load(ids_names_file)

