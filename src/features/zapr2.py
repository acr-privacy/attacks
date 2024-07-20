import base64
from pathlib import Path

import numpy as np
from keras.utils import pad_sequences, to_categorical
from sklearn.preprocessing import LabelEncoder

from data.util import SpeakersDataset, SpeechVsMusicDataset, WordsDataset


def preprocess_zapr2_speechvsmusic_dataset(p: Path, n_bits, seq_len):
    meta = SpeechVsMusicDataset.load_metadata(p)

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_zapr2_features(contents, n_bits, seq_len)

    y = LabelEncoder().fit_transform(meta.loc[:, "label"])
    y = to_categorical(y)

    return meta, x, y


def preprocess_zapr2_speaker_dataset(p: Path, n_bits, split, seq_len):
    meta = SpeakersDataset.load_metadata(p)
    split_idx = meta.loc[:, "split"] == split
    meta = meta.loc[split_idx, :]

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_zapr2_features(contents, n_bits, seq_len)

    y = LabelEncoder().fit_transform(meta.loc[:, "speaker_id"])
    y = to_categorical(y)

    return meta, x, y


def preprocess_zapr2_words_dataset(p: Path, n_bits, seq_len):
    samples_to_drop_idx = [
        # more then 4000 bit
        7304,
        7349,
        8280,
        19311,
        31673,
        31896,
        32162,
        34725,
        34891,
    ]
    meta = WordsDataset.load_metadata(p)
    meta.drop(meta.index[samples_to_drop_idx], axis=0, inplace=True)

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_zapr2_features(contents, n_bits, seq_len)
    y = LabelEncoder().fit_transform(meta.loc[:, "word"])
    y = to_categorical(y)
    return meta, x, y


def extract_zapr2_features(fingerprints, n_bits, seq_len):
    """
    Extract the bytes of each Zapr fingerprint (skip the first 20 bytes,
    i.e., the header), create their bitstring and group them into batches
    of `n_bits`.
    """
    bitwise_blocks = get_fingerprint_bitstrings(fingerprints)
    padded_sequences = pad_sequences(
        bitwise_blocks,
        padding="post",
        value=0,
        maxlen=seq_len * n_bits,
        dtype="int8",
    )
    x = padded_sequences.reshape(padded_sequences.shape[0], -1, n_bits)
    return x


def get_fingerprint_bitstrings(fingerprints):
    bitwise_blocks = []
    for fprint in fingerprints:
        byte_fprint = base64.b64decode(fprint)
        bitstring = ""
        for i, b in enumerate(byte_fprint[20:]):
            bitstring += f"{b:08b}"
        bitwise_blocks.append(
            np.array([bit for bit in bitstring], dtype=np.int8)
        )

    return bitwise_blocks


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
