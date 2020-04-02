import os
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy import signal
from scipy.signal import find_peaks

import Utility
from scipy.signal import butter, filtfilt
from Constants import global_sr, window_length, hop_size, window_length_2, hop_size_2, database_recordings

# Path to wavs
wav_path = "data" + os.path.sep + database_recordings
# Folder in which to store generated fingerprints so they don't have to be calculated everytime
generated = "data" + os.path.sep + "generated"


def compute_fingerprint(path):
    """
    Compute the fingerpringt for a given signal
    :param sig: The audio signal
    :return: The fingerprint
    """
    # Load signal and sample rate
    sig, sr = librosa.load(path, sr=None)

    # Convert signal to mono
    sig = librosa.core.to_mono(sig)

    # Resample to 22050Hz if generating fingerprint for query
    if sr != global_sr:
        sig = librosa.resample(sig, sr, global_sr)

    # Try bogus signal distortion
    # sig = np.concatenate((np.zeros(int(global_sr * 2.234)), sig))
    # mu, sigma = 0, 0.1
    # noise = np.random.normal(mu, sigma, sig.shape)
    # sig += noise
    # librosa.output.write_wav("data/asdf.wav", sig, global_sr)

    # Get mean and std of dataset
    mean, std = get_mean_std()

    # First STFT
    # Compute magnitude STFT with blackman window of 100ms and hop size of 25ms
    stft = np.abs(
        librosa.core.stft(sig, n_fft=window_length, hop_length=hop_size, win_length=window_length, window='blackman'))

    # Normalise by precalculated mean and std
    # stft = (stft - mean) / std

    # Convert power spectrogram to energy spectrogram
    stft = np.power(stft, 2)

    # Group the resulting frequency bands into 24 Bark bands
    # NOTE: We omit the 24th bark band as the lower cutoff frequency for the 24rd is 13500Hz and our global
    # SR is 22050Hz
    N_Barks = 23
    # Calculate Bark bands. This results in Z(b, m) (paper, page 2)
    spec_bark = np.zeros((N_Barks, stft.shape[1]))
    for i in range(stft.shape[0]):
        # Get the current "k" by mapping i to [0, global_sr / 2]
        k = Utility.mapFromTo(i, 0, stft.shape[0], 0, global_sr / 2)
        # Convert k to bark frequency
        k_bark = int(13 * np.arctan(k / 1315.8) + 3.5 * np.arctan(k / 7518.0))
        # Add energy to k_bark band
        spec_bark[k_bark] += stft[i]

    # Take six sub bands and convert to sone scale
    spec_bark = librosa.util.normalize(spec_bark[5:11])

    # Convert to sone scale
    # for row in range(spec_bark.shape[0]):
    #     for column in range(spec_bark.shape[1]):
    #         current = spec_bark[row, column]
    #         if current < 1:
    #             spec_bark[row, column] = 1
    #         elif 1 <= current <= 40:
    #             spec_bark[row, column] = 2 ** ((current - 40) / 10)
    #         else:
    #             spec_bark[row, column] = (current / 40) ** 2.642

    # Second STFT
    # Here a separate STFT for ONE SPECIFIC frequency "K" is calculated for each frequency band
    # Find onsets in terms of the small window (25ms) to determine starting positions for fingerprints
    onsets = get_onsets(np.copy(spec_bark))
    # onsets = np.arange(0, 320)

    # Get one fingerprint for each onset
    fps = []
    indices = []
    for onset in onsets:
        if onset + window_length_2 > spec_bark.shape[1]:
            break

        # Create subset of bark band spectrogram for current position as described in AudioPrint
        # (Ramona & Peeters, 2013)
        # This subset consists of 6 frequency bands around 1000Hz
        short_term_spectra = spec_bark[:, onset: onset + window_length_2]

        # This will contain a list of 4 fingerprints (1 every 0.5s)
        fingerprint = np.zeros((4, 6, 6))
        # For each short-term band k calculate the energies of 6 long-term bands
        for k in range(short_term_spectra.shape[0]):
            # Current print
            # Get band and normalise
            band = np.asfortranarray(librosa.util.normalize(short_term_spectra[k]))
            p = np.abs(librosa.core.stft(band,
                                         n_fft=window_length_2,
                                         hop_length=hop_size_2,
                                         win_length=window_length_2,
                                         # center=False,
                                         window=np.ones(window_length_2)))  # Rectangular window

            # Get frequency band corresponding to 2Hz
            band_2hz = int((2 / (hop_size_2 / 2)) * (window_length_2 / 2))
            band_range = p[band_2hz - 3 : band_2hz + 3]

            for frame in range(4):
                fingerprint[frame, k] = band_range[:, frame]

        # This results in a 4x36-dimensional concatenated vector
        for frame in range(fingerprint.shape[0]):
            frameTime = frame * hop_size_2
            if onset + frameTime in indices:
                continue
            fps.append(fingerprint[frame])
            indices.append(onset + frameTime)

    # Notify user
    print("Fingerprint generated for file: " + path)
    return fps, indices


def read_fingerprint(path):
    assert os.path.exists(
        generated), "Throw error if trying to read from 'generated' folder but the folder does not exist"
    fingerprint = np.load(path, allow_pickle=True)
    return fingerprint


def save_fingerprint(fingerprint, filename):
    # Create "generated" directory if not exists
    if not os.path.exists(generated):
        os.makedirs(generated)
    np.save(generated + os.path.sep + filename, fingerprint)


def get_mean_std():
    if os.path.exists("meanstd.npy"):
        meanstd = np.load("meanstd.npy")
        mean = meanstd[0]
        std = meanstd[1]
    else:
        # Get mean and std of whole dataset
        mean = 0
        std = 0
        counter = 0
        for wav in Path(wav_path).rglob("*.wav"):
            sys.stdout.write("\rCalculating STFT %i of 300" % counter)
            sys.stdout.flush()
            counter += 1
            # Load signal and sample rate
            sig, sr = librosa.load(wav, sr=None)

            # Convert signal to mono
            sig = librosa.core.to_mono(sig)

            # Resample to 22050Hz if generating fingerprint for query
            if sr != global_sr:
                sig = librosa.resample(sig, sr, global_sr)
            stft = np.abs(
                librosa.core.stft(sig, n_fft=window_length, hop_length=hop_size, win_length=window_length,
                                  window='blackman'))
            mean += np.mean(stft)
            std += np.std(stft)

        mean /= 300
        std /= 300
        np.save("meanstd", np.array([mean, std]))

    return mean, std


def get_onsets(bark):
    # Calculate E(m) for each m (frame time)
    Em = []
    bands = bark.shape[0]  # Will be 6
    frames = bark.shape[1]
    for m in range(frames):
        current = (1 / bands) * np.sum(bark[:, m])
        Em.append(current)

    Em = np.array(Em)
    # Normalise with a sliding window of 20 frames
    for i in range(0, Em.shape[0], 20):
        lookahead = Em.shape[0] - i if i + 20 > Em.shape[0] else 20
        current = Em[i: i + lookahead]
        Em[i: i + 20] = (current - np.median(current)) / np.std(current)

    # Create "Plateau" signal of 7 frames
    Pm = np.zeros(Em.shape[0])
    back = 7
    ahead = 7
    for i in range(Em.shape[0]):
        back = 0 if i - back < 0 else back
        ahead = ahead if i + ahead <= Em.shape[0] else Em.shape[0] - i

        Pm[i] = np.max(Em[i - back : i + ahead])

    # Create local maxima mask
    # Those are all frame positions for which Em and Pm are equal
    local_maxima = np.equal(Em, Pm)

    onsets = np.nonzero(local_maxima)[0]

    return onsets
