import os

import numpy as np
import librosa
import Utility
from Constants import global_sr, window_length, hop_size, window_length_2, hop_size_2

# Folder in which to store generated fingerprints so they don't have to be calculated everytime
generated = "data" + os.path.sep + "generated"

def compute_fingerprint(path):
    """
    Compute the fingerpringt for a given signal
    :param sig: The audio signal
    :return: The fingerprint
    """
    # Load signal and sample rate
    sig, sr = librosa.load(path)

    # Convert signal to mono
    sig = librosa.core.to_mono(sig)

    # Resample to 22050Hz
    if sr != global_sr:
        sig = librosa.resample(sig, sr, global_sr)

    # Normalise signal
    # sig *= 10 ** (96 / 20)

    # First STFT
    # Compute magnitude STFT with blackman window of 100ms and hop size of 25ms
    stft = np.abs(
        librosa.core.stft(sig, n_fft=window_length, hop_length=hop_size, win_length=window_length, window='blackman'))

    # Group the resulting frequency bands into 24 Bark bands
    # NOTE: We omit the 24th bark band as the lower cutoff frequency for the 24rd is 13500Hz and our global
    # SR is 22050Hz
    N_Barks = 23
    # Convert power spectrogram to energy spectrogram
    stft = np.power(stft, 2)
    # Calculate Bark bands. This results in Z(b, m) (paper, page 2)
    spec_bark = np.zeros((N_Barks, stft.shape[1]))
    for i in range(stft.shape[0]):
        # Get the current "k" by mapping i to [0, global_sr / 2]
        k = Utility.mapFromTo(i, 0, stft.shape[0], 0, global_sr / 2)
        # Convert k to bark frequency
        k_bark = int(13 * np.arctan(k / 1315.8) + 3.5 * np.arctan(k / 7518.0))
        # Add energy to k_bark band
        spec_bark[k_bark] += stft[i]

    # Apply Sone-scale
    for row in range(spec_bark.shape[0]):
        for column in range(spec_bark.shape[1]):
            if spec_bark[row, column] < 1:
                spec_bark[row, column] = 1
            elif 1 <= spec_bark[row, column] <= 40:
                spec_bark[row, column] = 2 ** ((spec_bark[row, column] - 40) / 10)
            elif spec_bark[row, column] > 40:
                spec_bark[row, column] = (spec_bark[row, column] / 40) ** 2.642
    # Second STFT
    # Here a separate STFT for ONE SPECIFIC frequency "K" is calculated for each frequency band
    # Test with K=2 TODO this might be bogus
    K = 2

    # Fingerprint
    fp = []
    for i in range(spec_bark.shape[0]):
        # for m in range(window_length_2 - 1):
        #     spec[m] = stft[i][m] * np.exp(1j * 2*np.pi * (K / window_length_2) * m)
        spec = np.abs(librosa.core.stft(np.asfortranarray(spec_bark[i]),
                                        n_fft=window_length_2,
                                        hop_length=hop_size_2,
                                        win_length=window_length_2,
                                        window=np.ones(window_length_2)))  # Rectangular window
        spec_k = spec[K]
        fp.append(spec_k)
    fp = np.array(fp)

    # Group adjacent frequencies together
    while fp.shape[0] > 36:
        new_rows = int(np.ceil(fp.shape[0] / 2))
        new_fp = np.zeros((new_rows, fp.shape[1]))
        for i in range(new_rows):
            if i * 2 + 1 >= fp.shape[0]:
                new_fp[i] = fp[i * 2]
            else:
                new_fp[i] = (fp[i * 2] + fp[i * 2 + 1]) / 2
        fp = new_fp

    # Notify user
    print("Fingerprint generated for file: " + path)
    # This results in a 33-dimensional concatenated vector (i.e. a matrix with shape (33, n))
    return fp

def read_fingerprint_from_csv(path):
    assert os.path.exists(generated), "Throw error if trying to read from 'generated' folder but the folder does not exist"
    fingerprint = np.loadtxt(path, delimiter=",")
    return fingerprint

def save_fingerprint_to_csv(fingerprint, filename):
    # Create "generated" directory if not exists
    if not os.path.exists(generated):
        os.makedirs(generated)
    np.savetxt(generated + os.path.sep + filename, fingerprint, delimiter=",")
