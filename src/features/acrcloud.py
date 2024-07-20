import base64
from pathlib import Path

import numpy as np
from keras.utils import pad_sequences, to_categorical
from sklearn.preprocessing import LabelEncoder

from data.util import SpeakersDataset, SpeechVsMusicDataset, WordsDataset


def preprocess_acrcloud_words_dataset(p: Path, seq_len):
    meta = WordsDataset.load_metadata(p)

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_feature_batches(contents, seq_len=seq_len)
    y = LabelEncoder().fit_transform(meta.loc[:, "word"])
    y = to_categorical(y)

    return meta, x, y


def preprocess_acrcloud_speaker_dataset(p: Path, split, seq_len):
    meta = SpeakersDataset.load_metadata(p)
    split_idx = meta.loc[:, "split"] == split
    meta = meta.loc[split_idx, :]

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_feature_batches(contents, seq_len=seq_len)
    y = LabelEncoder().fit_transform(meta.loc[:, "speaker_id"])
    y = to_categorical(y)

    return meta, x, y


def preprocess_acrcloud_speechvsmusic_dataset(p: Path, seq_len):
    meta = SpeechVsMusicDataset.load_metadata(p)

    contents = meta.loc[:, "path"].apply(read_content)
    x = extract_feature_batches(contents, seq_len=seq_len)

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


def extract_feature_batches(data, seq_len, skip_bytes=[]):
    """
    Extract the bitwise fingerprints with given parameters and reshape
    the features accordingly. They can be directly fed into an LSTM.
    """
    bits = 64 - (len(skip_bytes) * 8)
    x = get_bitwise_fingerprints(data, skip_bytes)

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


def get_bitwise_fingerprints(data, skip_bytes=None):
    """
    Given the base64 fingerprints, we transform each by converting each
    byte to binary and use the first max_bits bits as sequence. We pad
    them with zeros. Max_bits = 10000 could be around 8 to 9 seconds of
    rich audio.

    TODO: Should max_bits be adaptive to the maximum length of the computed
    bitstrings?

    skip_bytes can be a list of integers (0 to 7). The contained bytes in
    each 8-byte group are skipped.
    """
    if skip_bytes is None:
        skip_bytes = []

    bitwise_fprints = []
    for fprint in data:
        byte_fprint = base64.b64decode(fprint)
        bitstring = ""

        if skip_bytes:
            skip_indices = set()
            for b in skip_bytes:
                skip_indices |= set(np.arange(b, len(byte_fprint), 8))

        bitstring = "".join(
            (
                f"{b:08b}"
                for i, b in enumerate(byte_fprint)
                if (not skip_bytes) or (i not in skip_indices)
            )
        )
        bits = [bit for bit in bitstring]
        bits = np.array(bits, dtype=np.int8)

        bitwise_fprints.append(bits)
    return bitwise_fprints
