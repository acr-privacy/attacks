import base64
from pathlib import Path

import numpy as np
from keras.utils import pad_sequences, to_categorical
from sklearn.preprocessing import LabelEncoder

from data.util import SpeakersDataset, SpeechVsMusicDataset, WordsDataset


def preprocess_zapr1_speaker_dataset(p: Path, split, seq_len):
    meta = SpeakersDataset.load_metadata(p)
    split_idx = meta.loc[:, "split"] == split
    meta = meta.loc[split_idx, :]

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_zapr1_features(contents, seq_len=seq_len)

    y = LabelEncoder().fit_transform(meta.loc[:, "speaker_id"])
    y = to_categorical(y)

    return meta, x, y


def preprocess_zapr1_words_dataset(p: Path, seq_len):
    meta = WordsDataset.load_metadata(p)
    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_zapr1_features(contents, seq_len=seq_len)
    y = LabelEncoder().fit_transform(meta.loc[:, "word"])
    y = to_categorical(y)
    return meta, x, y


def preprocess_zapr1_speechvsmusic_dataset(p: Path, seq_len):
    meta = SpeechVsMusicDataset.load_metadata(p)

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_zapr1_features(contents, seq_len=seq_len)

    y = LabelEncoder().fit_transform(meta.loc[:, "label"])
    y = to_categorical(y)

    return meta, x, y


def extract_zapr1_features(fingerprints, seq_len, byte_indices=[0, 1, 2, 3]):
    """
    Extract bytes as bitstring from Zapr fingerprints. We skip the header
    of 20 bytes.

    :param fingerprints: Iterable that contains the fingerprints in base64
    encoded format.

    :param byte_indices: List of indices (0 to 3) that should be used of
    each 4 byte block of the fingerprint.
    """
    bitstrings = get_fingerprint_bitstrings(fingerprints, byte_indices)
    maxlen = seq_len * len(byte_indices) * 8
    padded = pad_sequences(
        bitstrings,
        padding="post",
        value=0,
        maxlen=maxlen,
        truncating="post",
        dtype="int8",
    )
    x = padded.reshape(padded.shape[0], -1, 8 * len(byte_indices))
    return x


def get_fingerprint_bitstrings(fingerprints, byte_indices=[0, 1, 2, 3]):
    bitstrings = []
    fprint_start = 20  # we start at byte index 20 (first one after the header)

    if len(byte_indices) == 0:
        raise Exception("Given byte indices must not be empty!")
    for fprint in fingerprints:
        byte_fprint = base64.b64decode(fprint)
        used_indices = []
        for b in byte_indices:
            used_indices += list(
                np.arange(fprint_start + b, len(byte_fprint), 4)
            )
        used_indices.sort()

        bitstring = ""
        for idx in used_indices:
            bitstring += f"{byte_fprint[idx]:08b}"
        bitstrings.append(np.array([bit for bit in bitstring], dtype=np.int8))

    return bitstrings


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
