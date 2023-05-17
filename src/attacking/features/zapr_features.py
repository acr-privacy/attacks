import base64
import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences


def extract_byte_blocks(fingerprints, blocksize=2):
    """
    Extract the bytes of each Zapr fingerprint (skip the first 20 bytes, i.e., the header), create their
    bitstring and group them into batches of `blocksize` bytes.
    """
    bitwise_blocks = []
    max_length = 0  # max. fingerprint length (in bit)
    for fprint in fingerprints:
        byte_fprint = base64.b64decode(fprint)
        max_length = max(max_length, len(byte_fprint) * 8)
        bitstring = ''
        for i, b in enumerate(byte_fprint[20:]):
            bitstring += f"{b:08b}"
        bitwise_blocks.append(np.array([bit for bit in bitstring], dtype=np.int8))
    # The length that we pad to must be divisible by 8 * blocksize
    padding_length = min([n for n in range(max_length, max_length + 8 * blocksize) if n % (8 * blocksize) == 0])
    padded_sequences = pad_sequences(bitwise_blocks, padding='post', value=0, maxlen=padding_length, truncating='post',
                                     dtype="int8")
    # Now reshape the sequences such that each one consists of 'blocksize * 8' bits
    x = padded_sequences.reshape(padded_sequences.shape[0], -1, blocksize * 8)
    return x


def extract_bitstrings(fingerprints, max_len):
    """
    Extract each fingerprint as a single long bitstring, suitable for feeding into a feed forward NN.
    Maxlen denotes the maximum number of bits that we pad (and possibly truncate) to.
    """
    bitstrings = []
    for fprint in fingerprints:
        byte_fprint = base64.b64decode(fprint)
        bitstring = ''
        for i, b in enumerate(byte_fprint[20:]):
            bitstring += f"{b:08b}"
        bitstrings.append(np.array([bit for bit in bitstring], dtype=np.int8))
    padded = pad_sequences(bitstrings, padding='post', value=0, maxlen=max_len, truncating='post', dtype='int8')
    return padded


def zapr_0_bytes(fingerprints, byte_indices):
    """
    Extract bytes as bitstring from Zapr0 fingerprints. We skip the header of 20 bytes.
    :param fingerprints: Iterable that contains the fingerprints in base64 encoded format.
    :param byte_indices: List of indices (0 to 3) that should be used of each 4 byte block of the fingerprint.
    """
    bitstrings = []
    fprint_start = 20  # we start at byte index 20 (first one after the header)

    if len(byte_indices) == 0:
        raise Exception("Given byte indices must not be empty!")
    for fprint in fingerprints:
        byte_fprint = base64.b64decode(fprint)
        used_indices = []
        for b in byte_indices:
            used_indices += list(np.arange(fprint_start + b, len(byte_fprint), 4))
        used_indices.sort()

        bitstring = ''
        for idx in used_indices:
            bitstring += f"{byte_fprint[idx]:08b}"
        bitstrings.append(np.array([bit for bit in bitstring], dtype=np.int8))
    maxlen = max([len(x) for x in bitstrings])
    padded = pad_sequences(bitstrings, padding='post', value=0, maxlen=maxlen, truncating='post', dtype='int8')
    x = padded.reshape(padded.shape[0], -1, 8 * len(byte_indices))
    return x


def extract_bytes3_4_integral(data):
    """
    Extract the integral values (as list) of bytes 3 & 4 of each Zapr (SC1 or WL2) fingerprint block.
    For each fingerprint, a list is returned. It is padded with 0 up to the max. length.
    """
    import base64
    from keras_preprocessing.sequence import pad_sequences

    X = []
    maxlen = 0
    for fprint in data["content"]:
        byte_fprint = base64.b64decode(fprint)
        freqs = []
        for i in range(22, len(byte_fprint), 4):
            freqs.append(int(byte_fprint[i]))
            freqs.append(int(byte_fprint[i+1]))
        X.append(freqs)
        maxlen = max(maxlen, len(freqs))
    padded = pad_sequences(X, maxlen=maxlen)
    return padded
