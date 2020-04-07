import os
from math import ceil

# Folder storing database recordings
database_recordings = "database_recordings"
query_recordings = "query_recordings"
generated = "generated"

# Path to wavs
wav_path = "data" + os.path.sep + database_recordings

# Name of stored classifier (k nearest neighbors trained with sklearn)
clf_name = "nearestNeighbor.joblib"


# Sampling rate used throughout the system
# NOTE: The query recordings have a SR of 44100Hz, so they will be resampled in the
# Fingerprint creation process
global_sr = 8000

# Shazam approach parameters
N_FFT = 2048
HOP_SIZE = 512

# Ramona approach parameters (unused)
# First STFT parameters
window_length = int(0.1 * global_sr)
hop_size = ceil(0.025 * global_sr)

first_stft_sr = 40

# Second STFT parameters
window_length_2 = 2 * first_stft_sr
hop_size_2 = int(0.5 * first_stft_sr)