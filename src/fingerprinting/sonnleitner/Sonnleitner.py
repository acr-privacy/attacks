import bisect
import itertools
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter

from audio.utils import resample
from common import hann, stft
from common.Peak import Peak

TARGET_FREQUENCY = 8000
SAMPLES_PER_FRAME = 1024
HOPSIZE = 32
OVERLAPPING_FRAMES = 992  # SAMPLES_PER_FRAME - HOPSIZE


# These two settings are described in the paper by Sonnleitner and Widmer (prefix "PAPER").
# However, they do not work well for short queries and/or reference tracks.
PAPER_REF_FINGERPRINT_SETTINGS = {
    'max_filter_width': 151,
    'max_filter_height': 75,
    'quads_per_second': 9,
    'grouping_center': 325,  # 1.3 seconds in the future
    'grouping_width': 200  # window is 800ms wide
}

PAPER_QUERY_FINGERPRINT_SETTINGS = {  # Depends on ref-settings. Calculated for params in paper and e_P = e_T = 20%.
    'max_filter_width': 125,
    'max_filter_height': 60,
    'quads_per_second': 1500,
    'grouping_center': 359,
    'grouping_width': 344
    # grouping windows starts 225/1.2 = 187 frames in future
    # grouping windows end 425/0.8 = 531 frames in future -> grouping center 359 frames in future
}

# We derived the following settings for short queries and reference tracks.
SHORT_REFERENCE_FINGERPRINT_SETTINGS = {
    'max_filter_width': 101,
    'max_filter_height': 51,
    'quads_per_second': 20,
    'grouping_center': 100,  # 0.4 seconds in the future
    'grouping_width': 170  # window is 680ms wide [+15 to +185 frames, so +60ms to +740ms]
}

# Computed via equations from the above values for tolerance bounds of 20%.
SHORT_QUERY_FINGERPRINT_SETTINGS = {
    'max_filter_width': 71,
    'max_filter_height': 35,
    'quads_per_second': 500,
    'grouping_center': 122,  # 0.488 seconds in the future
    'grouping_width': 210  # window is 840ms wide [+12 to +232 frames, so +48 ms to +928ms]
}


class Fingerprint:
    def __init__(self):
        self.hashes = []  # tuples (C'x, C'y, D'x, D'y)
        self.quads = []
        self.peaks = []


class Quad:
    def __init__(self, a: Peak, b: Peak, c: Peak, d: Peak):
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self._a_norm = None
        self._b_norm = None
        self._c_norm = None
        self._d_norm = None

    def is_valid(self):
        paper_constraint = self.A.y < self.B.y and \
                           self.A.x < self.C.x <= self.D.x <= self.B.x and \
                           self.A.y < self.C.y and \
                           self.D.y <= self.B.y

        # This seems to be missing in the paper. Otherwise, C and D do not have to lie in the AAP-rectangle
        own_constraint = self.C.y < self.B.y and self.A.y <= self.D.y
        return paper_constraint and own_constraint

    @property
    def peaks(self):
        return [self.A, self.B, self.C, self.D]

    def _compute_normalized_points(self):
        # Normalize points with respect to root point A of quad
        self._a_norm = (0, 0)
        self._b_norm = (self.B.x - self.A.x, self.B.y - self.A.y)
        self._c_norm = (self.C.x - self.A.x, self.C.y - self.A.y)
        self._d_norm = (self.D.x - self.A.x, self.D.y - self.A.y)

    def get_float_hash(self):
        """
        Compute the quad's hash as tuple (C'x,C'y,D'x,D'y) as float numbers (all in [0,1])
        """
        if not self._a_norm:
            self._compute_normalized_points()
        return (self._c_norm[0] / self._b_norm[0], self._c_norm[1] / self._b_norm[1],
                self._d_norm[0] / self._b_norm[0], self._d_norm[1] / self._b_norm[1])


def _extract_peaks(spectrum, fingerprint_settings):
    """
    Extracts a sorted list of peaks from the magnitude spectrum.
    Use a combination of max and min-filter (from image processing).
    Default values for max filter are taken from the paper (for reference quads).
    """

    # A filter of given size is centered on each field in the spectrum
    # and the max value of this window is put into the resulting array.
    maxima = maximum_filter(spectrum, size=(fingerprint_settings['max_filter_width'],
                                            fingerprint_settings['max_filter_height']))
    minima = minimum_filter(spectrum, size=(3, 3))

    # Possible peaks are detected at positions where the spec. value is equal to the window's max
    peak_candidates = maxima == spectrum
    #n_candidates = (peak_candidates == True).sum()
    #print(f"Found {n_candidates} candidate peaks before min filter")

    # Positions that are both max. and min. are e.g. completely silent regions. Do not use these as peaks
    max_neq_min = maxima != minima
    peaks = peak_candidates == max_neq_min

    #n_final_candidates = (peaks == True).sum()
    #print(f"Found {n_final_candidates} final peaks after min filter")

    # TODO: Clean up the peaks according to paper: Group peaks by magnitude and check for adjacent peaks in each group.
    # TODO: What is meant by 'parabolic interpolation' in the paper?
    peak_positions = np.argwhere(peaks)
    peaks = sorted([Peak(peak[1], peak[0]) for peak in peak_positions])
    return peaks


def _extract_quads(peaks, fingerprint_settings):
    """
    Extracts and returns a list of all valid quads from the (sorted!) peaks.
    The grouping region is centered grouping_center frames in the future and has width grouping_width.
    """
    grouping_center = fingerprint_settings['grouping_center']
    grouping_width = fingerprint_settings['grouping_width']
    quads = []
    for root_peak in peaks:
        # restrict candidate peaks so they only lie in the given window
        region_center = root_peak.x + grouping_center
        region_begin = region_center - int(grouping_width / 2)
        region_end = region_center + int(grouping_width / 2)

        # Peaks are sorted. Compute the slice that contains the possible pairing peaks
        # The y-value 1000 is arbitrary chosen. Has to be at least 512, cause we have 512 frequency bins
        left_idx = bisect.bisect_left(peaks, Peak(region_begin, 0))
        right_idx = bisect.bisect_right(peaks, Peak(region_end, 1000))
        candidate_peaks = peaks[left_idx:right_idx]

        # Because the peaks are sorted and a,c,d,b must be in x-ascending order for a valid quad,
        # combinations is sufficient (not permutations!). Combinations yields combinations in order.
        for c, d, b in itertools.combinations(candidate_peaks, 3):
            quad = Quad(root_peak, b, c, d)
            if quad.is_valid():
                quads.append(quad)

    #print(f"Extracted {len(quads)} valid quads")
    return quads


def _pick_strongest(quads, spectrogram, fingerprint_settings):
    """
    Sonnleitner and Widmer propose to limit the number of quads per second by only taking the n strongest ones.
    Strength is measured by magnitude of C and D.
    """
    quads_per_second = fingerprint_settings['quads_per_second']
    # First: group all quads based on in which second of the recording the peak A lies
    # One second (bin) corresponds to 250 frames (based on 8kHz and 4ms hop size)
    bin_size = 250
    max_frame = max(quads, key=lambda x: x.A.x).A.x

    n_bins = (max_frame // bin_size) + 1
    bins = [[] for _ in range(n_bins)]
    for quad in quads:
        target_bin = quad.A.x // bin_size
        bins[target_bin].append(quad)

    final_quads = []
    for bin in bins:
        # Pick the "strongest" (high sum of magnitude of C and D) quads of each bin according to paper.
        bin.sort(key=lambda q: spectrogram[q.C.y][q.C.x] + spectrogram[q.D.y][q.D.x], reverse=True)
        final_quads += bin[:quads_per_second]
    return final_quads


def compute_fingerprint(audio_stream, curr_sampling_rate, custom_settings=None, type='short_query'):
    """
    Compute the fingerprint and return the generated Fingerprint object.
    custom_settings can be a dict that contains settings for fingerprinting, similar to the existing ones.
    """
    if custom_settings:
        fingerprint_settings = custom_settings
    else:
        if type == 'paper_query':
            fingerprint_settings = PAPER_QUERY_FINGERPRINT_SETTINGS
        elif type == 'paper_reference':
            fingerprint_settings = PAPER_REF_FINGERPRINT_SETTINGS
        elif type == 'short_reference':
            fingerprint_settings = SHORT_REFERENCE_FINGERPRINT_SETTINGS
        elif type == 'short_query':
            fingerprint_settings = SHORT_QUERY_FINGERPRINT_SETTINGS
        else:
            print("Error: Invalid fingerprint type")
            return None
    # Resample the audio to the target 8kHz, then compute the STFT and take its magnitude
    resampled_audio = resample(audio_stream, curr_sampling_rate, TARGET_FREQUENCY)
    freqs, times, spectrum = stft(resampled_audio, fs=TARGET_FREQUENCY, nperseg=SAMPLES_PER_FRAME,
                                  window=hann(SAMPLES_PER_FRAME), noverlap=OVERLAPPING_FRAMES)
    spectrum = abs(spectrum)

    fingerprint = Fingerprint()
    fingerprint.peaks = _extract_peaks(spectrum, fingerprint_settings)
    valid_quads = _extract_quads(fingerprint.peaks, fingerprint_settings)
    if len(valid_quads) == 0:
        return None
    fingerprint.quads = _pick_strongest(valid_quads, spectrum, fingerprint_settings)

    for quad in fingerprint.quads:
        quad_hash = quad.get_float_hash()
        fingerprint.hashes.append(quad_hash)
    return fingerprint
