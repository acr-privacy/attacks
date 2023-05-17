import numpy as np
from scipy.io.wavfile import read


def readWAV(path):
    """
    Reads .wav file from the given path
    :param path: path of the .wav file
    :return: audio stream and sample rate
    """
    # read wav files
    # fs = sampling rate
    # data = audio stream
    fs, data = read(path)
    if data.ndim > 1:
        # Average all channels to one single channel.
        averages = np.mean(data, axis=1)
        data = np.array(averages, dtype=np.int16)

    return data, fs