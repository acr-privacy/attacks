from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

from attacking.features.sonnleitner_features import peak_bitstrings_feature
from attacking.common.metrics import f1, matthews_correlation_coefficient

import logging


def compute_bits_per_line(feature_mode):
    """
    Compute the number of needed input bits depending on the chosen feature. We consider every frequency peak as one
    unit, i.e., one step in the input sequence. Its frequency needs 9 bit (<=511). Time needs 10 bit for audio signals
    of at most 3 seconds. Both together need 19 bits.
    """
    if feature_mode == 'both':
        return 19
    elif feature_mode == 'time':
        return 10
    elif feature_mode == 'frequency':
        return 9
    else:
        raise Exception(f"Invalid feature mode {feature_mode}.")


def extract_batched_peak_bitstrings(data, mode):
    """
    Extract the peak bitstrings (ordered) from fingerprint and prepare them as batches.
    """
    x = peak_bitstrings_feature(data['content'], mode)
    print(f'x.shape={x.shape}')
    bits_per_line = compute_bits_per_line(mode)
    # import pdb; pdb.set_trace()
    try:
        x = x.reshape((x.shape[0], -1, bits_per_line))
    except ValueError as e:
        # TODO: Evil hack. Needs to be carefully checked why fingerprints are too long in some cases.
        mod_bits = x.shape[1] % bits_per_line
        if  mod_bits != 0:
            x = x[:, 0:-mod_bits]
            x = x.reshape((x.shape[0], -1, bits_per_line))
    return x


def build_speechvsmusic_model(lstm_units=64, lstm_activation='tanh', lstm_rec_do=0.4, dropout_after=0.4,
                              learning_rate=0.005, feature_mode='frequency'):
    """
    Build the model for differentiating speech and music based on given parameters.
    The default parameters were computed as result of a grid search.
    """
    bits_per_line = compute_bits_per_line(feature_mode)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(lstm_units, activation=lstm_activation, recurrent_dropout=lstm_rec_do))
    if dropout_after > 0:
        model.add(Dropout(dropout_after))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', 'AUC', Precision(), Recall(), matthews_correlation_coefficient, f1])
    return model


def build_speaker_recognition_model(
        n_classes,
        lstm_units=32, 
        lstm_activation='tanh',
        lstm_rec_do=0.4,
        dropout_after=0.2,
        learning_rate=0.005,
        feature_mode='both',
):
    """
    Build the model for recognizing 1 out of 40 speakers based on given parameters.
    The default parameters were computed as result of a grid search.
    """
    bits_per_line = compute_bits_per_line(feature_mode)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(lstm_units, activation=lstm_activation, recurrent_dropout=lstm_rec_do))
    if dropout_after > 0:
        model.add(Dropout(dropout_after))
    model.add(Dense(n_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])
    return model


def build_words_model(num_classes=10, lstm1_units=32, lstm1_rec_do=0.4, lstm2_units=32, lstm2_rec_do=0.2, lstm_activation='tanh',
                      dropout_after=0.2, learning_rate=0.005, feature_mode='both'):
    """
    Build the model for the 10 class words attack (numbers 0-9) based on given parameters.
    The default parameters were computed as result of a grid search.
    """
    bits_per_line = compute_bits_per_line(feature_mode)
    use_two_lstms = (lstm2_units > 0)
    model = Sequential()

    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(lstm1_units, recurrent_dropout=lstm1_rec_do, return_sequences=use_two_lstms,
                   activation=lstm_activation))
    if use_two_lstms:
        model.add(LSTM(lstm2_units, recurrent_dropout=lstm2_rec_do, activation=lstm_activation))
    if dropout_after > 0:
        model.add(Dropout(dropout_after))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    return model


def speechvsmusic_peakcount_classify(train_data, test_data):
    """
    Find the optimal (number of peaks) threshold on the training data that separates fingerprints of speech and music
    best (highest accuracy). Evaluate the accuracies on the test data with the found threshold.
    """
    from attacking.features.sonnleitner_features import get_peak_list
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
    import numpy as np

    def extract_n_peaks(data):
        lengths = []
        for X in data['content']:
            peaks = get_peak_list(X)
            lengths.append(len(peaks))
        return np.array(lengths).reshape((-1, 1))

    x_train = extract_n_peaks(train_data)

    min_n_peaks = int(min(x_train))
    max_n_peaks = int(max(x_train))

    accuracies = {}
    best_acc, best_threshold = 0, 0

    for threshold in range(min_n_peaks - 1, max_n_peaks + 1, 1):
        # everything >= threshold is class 0 (music). everything below: speech (class 1)
        preds = [1 if size < threshold else 0 for size in x_train]
        acc = accuracy_score(train_data['label'], preds)
        accuracies[threshold] = acc
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            print(f"New best threshold {threshold} bytes: Acc.: {acc}")

    print(f"On training data: Best threshold {best_threshold} with training accuracy {best_acc}.")
    # As result, use 79 peaks as threshold (less than this is speech, more is music)

    x_test = extract_n_peaks(test_data)
    test_preds = [1 if size < best_threshold else 0 for size in x_test]
    scores = {}
    scores['accuracy'] = accuracy_score(test_data['label'], test_preds)
    scores['f1'] = f1_score(test_data['label'], test_preds)
    scores['recall'] = recall_score(test_data['label'], test_preds)
    scores['precision'] = precision_score(test_data['label'], test_preds)
    scores['mcc'] = matthews_corrcoef(test_data['label'], test_preds)

    print(f'Scores on test data: {scores}')


# ------------------------------------
# The following methods are only for experimenting with different settings/ideas and not relevant for evaluation.
# ------------------------------------

def peak_words_attack(data, train=None, test=None):
    """
    Feed the sequence of peak frequencies of a fingerprint (each frequency 9 bit) into a RNN.
    This works with the new fingerprint format.
    After 75 epochs, we obtain 33% accuracy, after 220 epochs 35%.
    """
    print("Starting peak words attack! (using peak list)")
    feature_mode = 'both'
    if train is None or test is None:
        train, test = train_test_split(data, test_size=0.2, random_state=0, stratify=data['label'])
    bits_per_peak = compute_bits_per_line(feature_mode)

    x_train = extract_batched_peak_bitstrings(train, mode=feature_mode)
    x_test = extract_batched_peak_bitstrings(test, mode=feature_mode)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_peak)))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    opt = Adam(learning_rate=0.005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    print("Fit model on training data")
    history = model.fit(
        x_train,
        train['label'],
        batch_size=256,
        epochs=120,
        validation_data=(x_test, test['label'])
    )
    return model, history


def speech_vs_music_attack(train_data, val_data):
    """
    Differentiate speech and music. Works with the new fingerprint format. Seems to lead to around 97% accuracy.
    """
    mode = 'frequency'
    x_train = extract_batched_peak_bitstrings(train_data, mode=mode)
    x_val = extract_batched_peak_bitstrings(val_data, mode=mode)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, compute_bits_per_line(mode))))
    model.add(LSTM(32, recurrent_dropout=0.4))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print("Fit model on training data")
    history = model.fit(
        x_train,
        train_data['label'],
        batch_size=512,
        epochs=500,
        validation_data=(x_val, val_data['label']),
    )
    return model, history


def speaker_recognition_attack(data, train_data=None, val_data=None):
    """
    Recognize the exact speaker from a set of possible speakers.
    """
    feature_mode = 'both'
    if train_data is None or val_data is None:
        train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=1)

    logging.info(f"Training data has {len(train_data)} samples, test data has {len(val_data)} samples.")

    x_train = extract_batched_peak_bitstrings(train_data, feature_mode)
    x_val = extract_batched_peak_bitstrings(val_data, feature_mode)

    model = build_speaker_recognition_model(feature_mode=feature_mode)

    logging.info("Fit model on training data")
    history = model.fit(
        x_train,
        train_data['label'],
        batch_size=128,
        epochs=400,
        verbose=1,
        validation_data=(x_val, val_data['label'])
    )
    return model, history
