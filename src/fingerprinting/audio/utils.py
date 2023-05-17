import numpy as np
import samplerate


def resample(stream, curr_sampling_rate, target_sampling_rate, force_int16=False):

    """
    Resample audio. Idea from https://stackoverflow.com/questions/29085268/resample-a-numpy-array
    Note that scipy's resample seems badly suited for audio. We use this package instead.
    If force_int16, the samples are converted to int16 format, if resampling was applied.
    Note that resampling using samplerate library results in float32 values (Though they may not be in usual [-1,1] range).
    """
    if curr_sampling_rate != target_sampling_rate:
        resampling_factor = target_sampling_rate / curr_sampling_rate
        resampled = samplerate.resample(stream, resampling_factor, 'sinc_best')
        if force_int16:
            # Note that the result is (due to the used library) a float32 array.
            # Fix this, so libraries such as ACRCloud supports it.
            if resampled.dtype != np.int16:
                if min(resampled) < -1.0 or max(resampled) > 1.0:
                    # in this case, the audio is not in the usual range for floats. So: convert to int
                    resampled = resampled.astype(dtype=np.int16)
                else:
                    # the range is the correct float range of [-1,1]. Scale this up.
                    resampled = resampled * np.iinfo(np.int16).max
                    resampled = resampled.astype(dtype=np.int16)
        return resampled
    else:
        return stream
