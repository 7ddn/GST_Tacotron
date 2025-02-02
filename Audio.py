# https://github.com/keithito/tacotron/blob/master/util/audio.py
# https://github.com/carpedm20/multi-speaker-tacotron-tensorflow/blob/master/audio/__init__.py
# I only changed the hparams to usual parameters from oroginal code.

import numpy as np
from scipy import signal
import librosa.filters
import librosa


def preemphasis(x, preemphasis = 0.97):
    return signal.lfilter([1, -preemphasis], [1], x)

def inv_preemphasis(x, preemphasis = 0.97):
    return signal.lfilter([1], [1, -preemphasis], x)


def spectrogram(y, num_freq, hop_length, win_length, sample_rate, ref_level_db = 20, max_abs_value = None, spectral_subtract= False):    
    M = _magnitude(y, num_freq, hop_length, win_length, sample_rate, spectral_subtract)
    S = _amp_to_db(M) - ref_level_db
    return _normalize(S) if max_abs_value is None else _symmetric_normalize(S, max_abs_value= max_abs_value)

def inv_spectrogram(spectrogram, num_freq, hop_length, win_length, sample_rate, ref_level_db = 20, power = 1.5, max_abs_value = None, griffin_lim_iters= 60):
    '''Converts spectrogram to waveform using librosa'''
    spectrogram = _denormalize(spectrogram) if max_abs_value is None else _symmetric_denormalize(spectrogram, max_abs_value= max_abs_value)
    S = _db_to_amp(spectrogram + ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** power, num_freq, hop_length, win_length, sample_rate, griffin_lim_iters= griffin_lim_iters))          # Reconstruct phase

def melspectrogram(y, num_freq, hop_length, win_length, num_mels, sample_rate, max_abs_value = None, spectral_subtract= False):
    M = _magnitude(y, num_freq, hop_length, win_length, sample_rate, spectral_subtract)
    S = _amp_to_db(_linear_to_mel(M, num_freq, num_mels, sample_rate))
    return _normalize(S) if max_abs_value is None else _symmetric_normalize(S, max_abs_value= max_abs_value)

def spectrogram_and_mel(y, num_freq, hop_length, win_length, sample_rate, spect_ref_level_db = 20, num_mels= 80, max_abs_mels = None, spectral_subtract= False):
    M = _magnitude(y, num_freq, hop_length, win_length, sample_rate, spectral_subtract)
    spect_S = _normalize(_amp_to_db(M) - spect_ref_level_db)
    mel_S = _amp_to_db(_linear_to_mel(M, num_freq, num_mels, sample_rate))
    mel_S = _normalize(mel_S) if max_abs_mels is None else _symmetric_normalize(mel_S, max_abs_value= max_abs_mels)

    return spect_S, mel_S

def mfcc(y, num_freq, num_mfcc, hop_length, win_length, sample_rate, use_energy= False):
    n_fft = (num_freq - 1) * 2
    mfcc_Array = librosa.feature.mfcc(y, sr= sample_rate, n_mfcc= num_mfcc + 1, n_fft= n_fft, hop_length= hop_length, win_length= win_length)
    mfcc_Array = mfcc_Array[:-1] if use_energy else mfcc_Array[1:]
    
    return mfcc_Array

def _magnitude(y, num_freq, hop_length, win_length, sample_rate, spectral_subtract= False):
    D = _stft(preemphasis(y), num_freq, hop_length, win_length, sample_rate)
    M = np.abs(D)
    if spectral_subtract:
        M = np.clip(M - np.mean(M, axis= 1, keepdims= True) / 10, a_min= 0.0, a_max= np.inf)

    return M

def _griffin_lim(S, num_freq, hop_length, win_length, sample_rate, griffin_lim_iters = 60):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, num_freq, hop_length, win_length, sample_rate)

    for _ in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, num_freq, hop_length, win_length, sample_rate)))
        y = _istft(S_complex * angles, num_freq, hop_length, win_length, sample_rate)
    return y

def _stft(y, num_freq, hop_length, win_length, sample_rate):
    n_fft = (num_freq - 1) * 2
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y, num_freq, hop_length, win_length, sample_rate):
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _linear_to_mel(spectrogram, num_freq, num_mels, sample_rate):
    _mel_basis = _build_mel_basis(num_freq, num_mels, sample_rate)
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(num_freq, num_mels, sample_rate):
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sr = sample_rate, n_fft = n_fft, n_mels=num_mels)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S, min_level_db = -100):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _symmetric_normalize(S, min_level_db = -100, max_abs_value = 4):
    return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value, -max_abs_value, max_abs_value)

def _denormalize(S, min_level_db = -100):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _symmetric_denormalize(S, min_level_db = -100, max_abs_value = 4):
    return ((np.clip(S, -max_abs_value, max_abs_value) + max_abs_value) / (2 * max_abs_value) * -min_level_db) + min_level_db
