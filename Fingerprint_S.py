import hashlib
import os
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure
from skimage.feature import peak_local_max
from Constants import global_sr, N_FFT, HOP_SIZE, window_length_2, hop_size_2, database_recordings, wav_path

# Fan out
target_zone_x_dist = 1
peak_neighbourhood_size = 12
fan_out = 30

hash_time_delta_min = 1
hash_time_delta_max = 400


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

                if hash_time_delta_min <= t_diff <= hash_time_delta_max:
                    value = str(freq1) + "|" + str(freq2) + "|" + str(t_diff)
                    gen_hash = hashlib.sha1(value.encode('utf-8'))
                    # Append hash and offset
                    hashes.append((gen_hash, t1))

    return hashes

def detect_peaks(spec):
    """
    Take a spectrogram and compute local peaks
    """

    struct = generate_binary_structure(2, 2)
    neighborhood = iterate_structure(struct, peak_neighbourhood_size)

    # Create a mask using the maximum_filter
    # local_max = maximum_filter(spec, footprint=neighborhood) == spec

    # skimage approach
    freq_time = peak_local_max(spec, footprint=neighborhood)
    time = freq_time[:, 1]
    freq = freq_time[:, 0]

    # Create a background mask
    #background = (spec < background_threshold)
    # Use binary erosion to remove border lines
    #eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    # Get final peak mask using XOR
    #detected_peaks = local_max ^ eroded_background
    # Extract peak values and indices
    #values = spec[detected_peaks]
    #freq, time = np.where(detected_peaks)
    # Filter peaks using amplitude thresholding
    # peaks = zip(time, freq, values)
    # peaks_filtered = [x for x in peaks if x[2] > min_amp]
    # frames = np.array([x[0] for x in peaks_filtered])
    # frequencies = np.array([x[1] for x in peaks_filtered])

    frames = time
    frequencies = freq
    print("Number of peaks found: " + str(len(frames)))

    return frames, frequencies

