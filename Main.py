import numpy as np
import Fingerprint
import Database
import os
from Constants import clf_name

# Note: The database recordings have a 22050 Hz sampling rate,
# while the query set recordings have a 44100 Hz sampling rate

# Initialise database
Database.initialise()

# Try to query a fingerprint
# Generate fingerprint from query set
fp_query = Fingerprint.compute_fingerprint("data/query_recordings/classical.00000-snippet-10-20.wav")

# Train classifier if it doesn't exist yet
# if not os.path.exists(clf_name):
clf = Database.train_classifier()
Database.save_classifier(clf)

Database.query(fp_query)


# Generate a fingerprint
path = "data/database_recordings/classical.00000.wav"
# fingerprint = Fingerprint.compute_fingerprint(path)

