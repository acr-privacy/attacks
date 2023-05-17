import os
import re
import pandas as pd
import logging
from glob import glob


def balance_classes(df: pd.DataFrame, n_per_class: int):
    """
    Balance the classes. Selects n rows of df for each class (label).
    """
    counts = df['label'].value_counts()
    n_samples = min(min(counts), n_per_class)
    # since we can only discard samples, we pick n_samples samples of each class
    limited = df.groupby(['label']).apply(lambda grp: grp.sample(n=n_samples)).reset_index(level=[0, 1], drop=True)
    return limited, n_samples


def limit_classes(df: pd.DataFrame, n_classes: int):
    """
    Limit the number of classes to a given number. We take the first different 'n_classes' classes that we encounter.
    """
    unique_labels = df['label'].unique()
    selected_labels = unique_labels[:n_classes]
    filtered_data = df.loc[df['label'].isin(selected_labels)]
    return filtered_data, len(selected_labels)


def extract_file_contents(folder: str, mode: str = 'number'):
    """
    Extract the file contents for each .fprint file in the given folder.
    Build dataframe of file name, file content and label (as string).
    'mode' can be either 'number' (for words recognition of english digits), 'speaker_librispeech' (for speaker
    recognition from the librispeech dataset) or 'speech_vs_music' (for the differentiation of speech and music).
    The label is determined either based on the filename (number and speaker recognition)
    or the complete filepath (speech vs music).
    """
    filepaths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.fprint'))]
    data = {'filename': [], 'content': [], 'label': []}

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        if mode == 'number':
            # extract the number ('one', 'two', ... from the filename). Filename: 1234_eight.fprint
            # labeling scheme Moritz: r'[0-9a-f]+-([a-zA-Z]+)-[0-9]+\.fprint'
            label = re.match(r'[0-9a-f]+-([a-zA-Z]+)-[0-9]+\.fprint', filename).group(1)
            data['label'].append(label)
        elif mode == 'speaker_librispeech':
            # The Speaker ID is the number coming before the first dash (right at the beginning)
            speaker_id = re.match(r'([0-9]+)-[0-9]+-[0-9]+((-chunk){0,1}-[0-9]+)?.fprint', filename).group(1)
            data['label'].append(speaker_id)
        elif mode == 'speech_vs_music':
            # It is expected that the filepath of music files (class 0) contains "fma" (from Free Music Archive)
            # and that the filepath of any speech file (class 1) does not contain "fma".
            data['label'].append(0 if 'fma' in filepath else 1)
        data['filename'].append(filename)
        with open(filepath, 'r') as f:
            # Load file content and skip BEGIN and END line
            content = ' '.join([l.strip() for l in f.readlines() if l.strip() != 'BEGIN' and l.strip() != 'END'])
        data['content'].append(content)
    df = pd.DataFrame(data)
    return df


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
