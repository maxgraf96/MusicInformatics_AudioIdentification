import hashlib
import os
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from skimage.feature import peak_local_max

from Constants import global_sr, N_FFT, HOP_SIZE

# Fan out
target_zone_x_dist = 1
peak_neighbourhood_size = 12
fan_out = 40

hash_time_delta_min = 1
hash_time_delta_max = 400

# Specifies the "height" of the window used for neighbour search during hashing
# A value of 80 means that neighbors with frequencies in the range current - 80 to current + 80
# will be considered
freq_margin = 80


def compute_fingerprint(path):
    """
       Compute the fingerpringt for a given signal
       :param sig: The audio signal
       :return: The fingerprint
       """

    # Store spectrogram if not there
    filename = str(path).split(os.path.sep)[-1]
    specpath = "data" + os.path.sep + "spec" + os.path.sep + filename
    # Create folder for saved spectrograms if not exists
    if not os.path.exists("data" + os.path.sep + "spec"):
        Path("data" + os.path.sep + "spec").mkdir(parents=True, exist_ok=True)
    if os.path.exists(specpath + ".npy"):
        stft = np.load(specpath + ".npy")
    else:
        # Load signal and sample rate
        sig, sr = librosa.load(path, sr=None)

        # Convert signal to mono
        sig = librosa.core.to_mono(sig)

        # Resample to global_sr (8000Hz)
        if sr != global_sr:
            sig = librosa.resample(sig, sr, global_sr)

        stft = np.abs(librosa.core.stft(sig,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE,
                                        window='hann',
                                        pad_mode='constant'
                                        )) ** 2

        # Convert to dB scale
        stft = librosa.core.power_to_db(stft)
        np.save(specpath, stft)

    # Calculate peak times (in stft frames and their corresponding frequencies)
    frames, freqs = detect_peaks(stft)

    # Generate hash
    hashes = create_hash(frames, freqs)

    print("Created fingerprint for " + str(path))
    print("Number of hashes found: " + str(len(hashes)))
    print()

    return hashes

def create_hash(frames, freqs):
    # Brute force hash generation by comparing each peak to its neighbors in the zone regulated
    # By the fan value
    hashes = []
    for i in range(frames.size):
        for j in range(target_zone_x_dist, fan_out):
            if i + j < frames.size:
                freq1 = freqs[i]
                freq2 = freqs[i + j]

                t1 = frames[i]
                t2 = frames[i + j]

                # Get time difference
                t_diff = t2 - t1

                if hash_time_delta_min <= t_diff <= hash_time_delta_max \
                        and freq2 - freq_margin <= freq1 <= freq2 + freq_margin:
                    value = str(freq1) + "|" + str(freq2) + "|" + str(t_diff)
                    gen_hash = hashlib.sha1(value.encode('utf-8'))
                    # Append hash and offset
                    hashes.append((gen_hash, t1))

    return hashes

def detect_peaks(spec):
    """
    Take a spectrogram and compute local peaks
    """

    # Binary structure defining the neighborhood
    struct = generate_binary_structure(2, 2)
    neighborhood = iterate_structure(struct, peak_neighbourhood_size)

    # skimage approach to find 2d local peaks
    freq_time = peak_local_max(spec, footprint=neighborhood)
    time = freq_time[:, 1]
    freq = freq_time[:, 0]

    frames = time
    frequencies = freq
    print("Number of peaks found: " + str(len(frames)))

    return frames, frequencies

