from math import ceil

# Folder storing database recordings
database_recordings = "database_recordings"
generated = "generated"

# Name of stored classifier (k nearest neighbors trained with sklearn)
clf_name = "nearestNeighbor.joblib"


# Sampling rate used throughout the system
# NOTE: The query recordings have a SR of 44100Hz, so they will be resampled in the
# Fingerprint creation process
global_sr = 22050
first_stft_sr = 40

# First STFT parameters
window_length = 2048
hop_size = ceil(0.025 * global_sr)

# Second STFT parameters
window_length_2 = 2 * first_stft_sr
hop_size_2 = int(0.5 * first_stft_sr)