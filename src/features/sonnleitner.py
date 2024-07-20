from pathlib import Path

import numpy as np
from keras.utils import pad_sequences, to_categorical
from sklearn.preprocessing import LabelEncoder

from data.util import SpeakersDataset, SpeechVsMusicDataset, WordsDataset


def preprocess_sonnleitner_words_dataset(p: Path, seq_len):
    samples_to_drop_idx = [
        # all zero subfingerprints at beginning
        20192,
        24076,
        # longer than 48 sub fingerprints
        10389,
        13130,
        16851,
        20883,
    ]

    meta = WordsDataset.load_metadata(p)
    meta.drop(meta.index[samples_to_drop_idx], axis=0, inplace=True)

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_batched_peak_bitstrings(contents, seq_len=seq_len)
    y = LabelEncoder().fit_transform(meta.loc[:, "word"])
    y = to_categorical(y)

    return meta, x, y


def preprocess_sonnleitner_speaker_dataset(p: Path, split, seq_len):
    meta = SpeakersDataset.load_metadata(p)
    split_idx = meta.loc[:, "split"] == split
    meta = meta.loc[split_idx, :]

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_batched_peak_bitstrings(contents, seq_len=seq_len)
    y = LabelEncoder().fit_transform(meta.loc[:, "speaker_id"])
    y = to_categorical(y)

    return meta, x, y


def preprocess_sonnleitner_speechvsmusic_dataset(p: Path, seq_len):
    samples_to_drop_idx = [
        # longer than 180 sub fingerprints
        7122,
        29024,
    ]
    meta = SpeechVsMusicDataset.load_metadata(p)
    meta.drop(meta.index[samples_to_drop_idx], axis=0, inplace=True)

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_batched_peak_bitstrings(contents, seq_len=seq_len)

    y = LabelEncoder().fit_transform(meta.loc[:, "label"])
    y = to_categorical(y)

    return meta, x, y


def read_content(p: str) -> str:
    with Path(p).open("r") as f:
        content = " ".join(
            [
                ln.strip()
                for ln in f.readlines()
                if (ln.strip() != "BEGIN") and (ln.strip() != "END")
            ]
        )

    return content


def extract_batched_peak_bitstrings(data, seq_len, mode="both"):
    """
    Extract the peak bitstrings (ordered) from
    fingerprint and prepare them as batches.
    """
    bits = compute_bits_per_line(mode)
    x = peak_bitstrings_feature(data, mode)
    x = pad_sequences(
        x,
        padding="post",
        truncating="post",
        maxlen=bits * seq_len,
        value=0,
        dtype="int8",
    )
    x = x.reshape((x.shape[0], -1, bits))
    return x


def compute_bits_per_line(feature_mode):
    """
    Compute the number of needed input bits depending on the chosen feature.
    We consider every frequency peak as one unit, i.e., one step in the input
    sequence. Its frequency needs 9 bit (<=511). Time needs 10 bit for audio
    signals of at most 3 seconds. Both together need 19 bits.
    """
    if feature_mode == "both":
        return 19
    elif feature_mode == "time":
        return 10
    elif feature_mode == "frequency":
        return 9
    else:
        raise Exception(f"Invalid feature mode {feature_mode}.")


def peak_bitstrings_feature(fingerprints, feature_mode="both"):
    """
    Extract all the found peaks (second part of the fingerprint) and
    concatenate either time frame or frequencies or both of all peaks to one
    single bitstring. Pad up to maximum length. Use 10 bits per time frame and
    9 bits per frequency (sufficient for 512 bins).
    """
    fprint_bitstrings = []
    for fprint in fingerprints:
        bitstring = ""
        for peak in get_peak_list(fprint):
            if feature_mode == "time":
                bitstring += f"{peak.x:010b}"
            elif feature_mode == "frequency":
                bitstring += f"{peak.y:09b}"
            else:
                bitstring += f"{peak.x:010b}{peak.y:09b}"
        fprint_bitstrings.append(
            np.array([bit for bit in bitstring], dtype=np.int8)
        )
    return fprint_bitstrings


def get_peak_list(fingerprint):
    """
    Extract list of all the (sorted) peaks from the second part of
    the fingerprint.
    """
    peaks = []
    peak_section = fingerprint.split("  ")[1]
    for line in peak_section.split(" "):
        peaks.append(_peak_from_line(line))
    return peaks


def _peak_from_line(line):
    """
    Create Peak object from a line. The line should contain
    the time frame and the frequency bin, separated by ','.
    """
    coords = line.split(",")
    return Peak(int(coords[0]), int(coords[1]))


class Peak:
    def __init__(self, x: int, y: int):
        self.x = x  # Frame number
        self.y = y  # Frequency bin

    def __eq__(self, other):
        if not isinstance(other, Peak):
            return False
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __str__(self):
        return f"({self.x},{self.y})"

    def __hash__(self):
        return hash((self.x, self.y))
