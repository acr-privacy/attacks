import base64
import numpy as np
from keras_preprocessing.sequence import pad_sequences


def get_bitwise_fingerprints(fingerprints, max_bits=10000, skip_bytes=None, pad_value=0):
    """
    Given the base64 fingerprints, we transform each by converting each byte to binary and use the first
    max_bits bits as sequence. We pad them with zeros.
    Max_bits = 10000 could be around 8 to 9 seconds of rich audio.
    TODO: Should max_bits be adaptive to the maximum length of the computed bitstrings?
    skip_bytes can be a list of integers (0 to 7). The contained bytes in each 8-byte group are skipped.
    """
    if skip_bytes is None:
        skip_bytes = []
    bitwise_fprints = []

    for fprint in fingerprints:
        byte_fprint = base64.b64decode(fprint)
        bitstring = ''
        if skip_bytes:
            skip_indices = set()
            for b in skip_bytes:
                skip_indices |= set(np.arange(b, len(byte_fprint), 8))

        for i, b in enumerate(byte_fprint):
            if skip_bytes and i in skip_indices:
                continue
            bitstring += f"{b:08b}"
        bitwise_fprints.append(np.array([bit for bit in bitstring[:max_bits]], dtype=np.int8))
    padded_sequences = pad_sequences(bitwise_fprints, padding='post', value=pad_value, maxlen=max_bits,
                                     truncating='post')
    return padded_sequences
