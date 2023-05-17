"""
Config file that can be imported and contains the paths to the datasets, labelencoders
and to the fingerprint and experiment root folder.
"""
import os

FINGERPRINT_ROOT = os.environ['FINGERPRINT_ROOT']
EXPERIMENT_ROOT = os.environ['EXPERIMENT_ROOT']
_LABELENCODER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labelencoders')

CV_RUNS = 10

WORDS_LABELENCODER = os.path.join(_LABELENCODER_DIR, 'speechcommands_words_35_labelencoder.joblib')

SPEAKER_TRAIN_LABELENCODER = os.path.join(_LABELENCODER_DIR, 'librispeech_train_100_speaker_40speakers_labelencoder.joblib')
SPEAKER_LIBRISPEECH_TEST_LABELENCODER = os.path.join(_LABELENCODER_DIR, 'librispeech_train_100_speaker_40speakers_test_labelencoder.joblib')
SPEAKER_YOUTUBE_TEST_LABELENCODER = os.path.join(_LABELENCODER_DIR, 'youtube_train_100_speaker_50speakers_test_labelencoder.joblib')

DATAPATHS = {
    # Speech vs Music collections
    'SONNLEITNER_SPEECHVSMUSIC_TRAIN': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/sonnleitner/training/'),
    'SONNLEITNER_SPEECHVSMUSIC_VAL': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/sonnleitner/validation/'),
    'SONNLEITNER_SPEECHVSMUSIC_TEST': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/sonnleitner/testing/'),

    'ACRCLOUD_SPEECHVSMUSIC_TRAIN': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/acrcloud/training/'),
    'ACRCLOUD_SPEECHVSMUSIC_VAL': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/acrcloud/validation/'),
    'ACRCLOUD_SPEECHVSMUSIC_TEST': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/acrcloud/testing/'),

    'ZAPR_ALG1_SPEECHVSMUSIC_TRAIN': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/zapr_alg1/training/'),
    'ZAPR_ALG1_SPEECHVSMUSIC_VAL': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/zapr_alg1/validation/'),
    'ZAPR_ALG1_SPEECHVSMUSIC_TEST': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/zapr_alg1/testing/'),

    'ZAPR_ALG2_SPEECHVSMUSIC_TRAIN': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/zapr_alg2/training/'),
    'ZAPR_ALG2_SPEECHVSMUSIC_VAL': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/zapr_alg2/validation/'),
    'ZAPR_ALG2_SPEECHVSMUSIC_TEST': os.path.join(FINGERPRINT_ROOT, 'speechvsmusic/zapr_alg2/testing/'),


    # Words (digits) dataset
    'SONNLEITNER_WORDS_TRAINING': os.path.join(FINGERPRINT_ROOT, 'words/sonnleitner/training/'),
    'SONNLEITNER_WORDS_TESTING': os.path.join(FINGERPRINT_ROOT, 'words/sonnleitner/testing/'),

    'ACRCLOUD_WORDS_TRAINING': os.path.join(FINGERPRINT_ROOT, 'words/acrcloud/training/'),
    'ACRCLOUD_WORDS_TESTING': os.path.join(FINGERPRINT_ROOT, 'words/acrcloud/testing/'),

    'ZAPR_ALG1_WORDS_TRAINING': os.path.join(FINGERPRINT_ROOT, 'words/zapr_alg1/training/'),
    'ZAPR_ALG1_WORDS_TESTING': os.path.join(FINGERPRINT_ROOT, 'words/zapr_alg1/testing/'),

    'ZAPR_ALG2_WORDS_TRAINING': os.path.join(FINGERPRINT_ROOT, 'words/zapr_alg2/training/'),
    'ZAPR_ALG2_WORDS_TESTING': os.path.join(FINGERPRINT_ROOT, 'words/zapr_alg2/testing/'),


    # Speaker Recognition 40 Speakers
    'SONNLEITNER_SPEAKER_RECOGNITION_TRAIN': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/sonnleitner/training/'),
    'ACRCLOUD_SPEAKER_RECOGNITION_TRAIN': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/acrcloud/training/'),
    'ZAPR_ALG1_SPEAKER_RECOGNITION_TRAIN': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/zapr_alg1/training/'),
    'ZAPR_ALG2_SPEAKER_RECOGNITION_TRAIN': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/zapr_alg2/training/'),

    'ACRCLOUD_LIBRISPEECH_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/acrcloud/testing/'),
    'SONNLEITNER_LIBRISPEECH_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/sonnleitner/testing/'),
    'ZAPR_ALG1_LIBRISPEECH_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/zapr_alg1/testing/'),
    'ZAPR_ALG2_LIBRISPEECH_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'librispeech_40speakers/zapr_alg2/testing/'),

    'SONNLEITNER_YOUTUBE_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'youtube_50speakers/sonnleitner/testing/'),
    'ACRCLOUD_YOUTUBE_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'youtube_50speakers/acrcloud/testing/'),
    'ZAPR_ALG1_YOUTUBE_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'youtube_50speakers/zapr_alg1/testing/'),
    'ZAPR_ALG2_YOUTUBE_SPEAKER_RECOGNITION_TEST': os.path.join(FINGERPRINT_ROOT, 'youtube_50speakers/zapr_alg2/testing/'),
}

