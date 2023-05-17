from keras_preprocessing.sequence import pad_sequences
from fingerprinting.common.Peak import Peak
import numpy as np


def _peak_from_line(line):
    """
    Create Peak object from a line. The line should contain the time frame and the frequency bin, separated by ','.
    """
    coords = line.split(',')
    return Peak(int(coords[0]), int(coords[1]))


def get_peak_list(fingerprint):
    """
    Extract list of all the (sorted) peaks from the second part of the fingerprint.
    """
    peaks = []
    peak_section = fingerprint.split('  ')[1]
    for line in peak_section.split(' '):
        peaks.append(_peak_from_line(line))
    return peaks


def peak_bitstrings_feature(fingerprints, feature_mode='both'):
    """
    Extract all the found peaks (second part of the fingerprint) and
    concatenate either time frame or frequencies or both of all peaks to one single bitstring. Pad up to maximum length.
    Use 10 bits per time frame and 9 bits per frequency (sufficient for 512 bins).
    """
    fprint_bitstrings = []
    for fprint in fingerprints:
        bitstring = ''
        for peak in get_peak_list(fprint):
            if feature_mode == 'time':
                bitstring += f"{peak.x:010b}"
            elif feature_mode == 'frequency':
                bitstring += f"{peak.y:09b}"
            else:
                bitstring += f"{peak.x:010b}{peak.y:09b}"
        fprint_bitstrings.append(np.array([bit for bit in bitstring], dtype=np.int8))
    max_length = max([len(bitstring) for bitstring in fprint_bitstrings])
    final_sequences = pad_sequences(fprint_bitstrings, maxlen=max_length, padding='post', value=0, dtype=np.int8,
                                    truncating='post')
    return final_sequences
