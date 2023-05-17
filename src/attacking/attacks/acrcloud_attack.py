import os

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

import attacking.common.config as config
from attacking.common.metrics import f1, matthews_correlation_coefficient
from attacking.features.acrcloud_features import get_bitwise_fingerprints
import numpy as np
import logging
import base64


def extract_feature_batches(data, skip_bytes, max_bits):
    """
    Extract the bitwise fingerprints with given parameters and reshape the features accordingly.
    They can be directly fed into an LSTM.
    """
    bits_per_line = 64 - (len(skip_bytes) * 8)
    x = get_bitwise_fingerprints(data['content'], max_bits, skip_bytes, pad_value=0)
    x = x.reshape((x.shape[0], max_bits // bits_per_line, bits_per_line))
    return x


def build_speechvsmusic_model(lstm_units=32, lstm_activation='tanh', lstm_rec_do=0.4, dropout_after=0.2,
                              learning_rate=0.005, bits_per_line=32):
    """
    Build the model for differentiating speech and music based on given parameters.
    The default parameters were computed as result of a grid search.
    """
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(lstm_units, activation=lstm_activation, input_shape=(None, bits_per_line),
                   recurrent_dropout=lstm_rec_do))
    if dropout_after > 0:
        model.add(Dropout(dropout_after))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', 'AUC', Precision(), Recall(), matthews_correlation_coefficient, f1])
    return model


def build_words_model(num_classes=10, lstm1_units=32, lstm1_rec_do=0.4, lstm2_units=32, lstm2_rec_do=0.2, lstm_activation='tanh',
                      dropout_after=0.2, learning_rate=0.001, bits_per_line=48):
    """
    Build the model for the 10 class words attack (numbers 0-9) based on given parameters.
    The default parameters were computed as result of a grid search.
    """
    use_two_lstms = (lstm2_units > 0)
    model = Sequential()

    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(lstm1_units, input_shape=(None, bits_per_line), recurrent_dropout=lstm1_rec_do,
                   return_sequences=use_two_lstms, activation=lstm_activation))
    if use_two_lstms:
        model.add(LSTM(lstm2_units, recurrent_dropout=lstm2_rec_do, activation=lstm_activation))
    if dropout_after > 0:
        model.add(Dropout(dropout_after))

    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    return model


def build_speaker_recognition_model(
        n_classes,
        lstm_units=32, 
        lstm_activation='tanh', 
        lstm_rec_do=0.4,
        dropout_after=0.2,
        learning_rate=0.001,
        bits_per_line=32
):
    """
    Build the model for recognizing 1 out of 40 speakers based on given parameters.
    The default parameters were computed as result of a grid search.
    """
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(lstm_units, activation=lstm_activation, input_shape=(None, bits_per_line),
                   recurrent_dropout=lstm_rec_do))
    if dropout_after > 0:
        model.add(Dropout(dropout_after))
    model.add(Dense(n_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])
    return model


def speechvsmusic_fprint_size_classify(train_data, test_data):
    """
    Find the optimal (fingerprint size) threshold on the training data that separates fingerprints of speech and music
    best (highest accuracy). Evaluate the accuracies on the test data with the found threshold.
    """
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef

    def extract_fingerprint_length(data):
        lengths = []
        for X in data['content']:
            byte_fprint = base64.b64decode(X)
            lengths.append(len(byte_fprint))
        return np.array(lengths).reshape((-1, 1))

    x_train = extract_fingerprint_length(train_data)

    min_fprint_size = int(min(x_train))
    max_fprint_size = int(max(x_train))

    accuracies = {}
    best_acc, best_threshold = 0, 0
    for threshold in range(min_fprint_size - 4, max_fprint_size + 4, 8):
        # everything >= threshold is class 0 (music). everything below: speech (class 1)
        preds = [1 if size < threshold else 0 for size in x_train]
        acc = accuracy_score(train_data['label'], preds)
        accuracies[threshold] = acc
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            print(f"New best threshold {threshold} bytes: Acc.: {acc}")

    print(f"On training data: Best threshold {best_threshold} with training accuracy {best_acc}.")
    # As result, use 580 byte as threshold (less than this is speech, more is music)

    x_test = extract_fingerprint_length(test_data)
    test_preds = [1 if size < best_threshold else 0 for size in x_test]
    scores = {}
    scores['accuracy'] = accuracy_score(test_data['label'], test_preds)
    scores['f1'] = f1_score(test_data['label'], test_preds)
    scores['recall'] = recall_score(test_data['label'], test_preds)
    scores['precision'] = precision_score(test_data['label'], test_preds)
    scores['mcc'] = matthews_corrcoef(test_data['label'], test_preds)

    print(f'Scores on test data: {scores}')
    #dump(scores, f'acrcloud_speechvsmusic_evaluation_results_threshold_{best_threshold}bytes.joblib')

# ------------------------------------
# The following methods are only for experimenting with different settings/ideas and not relevant for evaluation.
# ------------------------------------


def speech_vs_music_attack(train_data, val_data, experiment_name, dataset='fma-librispeech-train-100', use_callbacks=True):
    callbacks = []
    if use_callbacks:
        logging.info(f"Starting experiment with name {experiment_name}")
        output_root = os.path.join(config.EXPERIMENT_ROOT, f'{dataset}_speech_vs_music/deezer_enc/', experiment_name)
        model_path = os.path.join(output_root, 'models')
        if os.path.exists(output_root):
            logging.error(f"Experiment folder {output_root} already exists. Aborting.")
            return None
        os.makedirs(model_path)

        tensorboard_callback = TensorBoard(log_dir=os.path.join(output_root, 'tb'), histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(model_path,
                                  "model.{epoch:02d}-valloss{val_loss:.2f}-valauc{val_auc:.2f}-valacc{val_accuracy:.2f}.hdf5"),
            monitor='val_loss')
        callbacks = [
            tensorboard_callback,
            checkpoint_callback
        ]

    skip_bytes = [2, 3, 4, 5, 6, 7]  # Only Bytes 0 & 1 are enough for good results
    bits_per_line = 64 - (len(skip_bytes) * 8)

    max_bits = 1680  # This is enough for 3 seconds of audio, if only Bytes 0 & 1 are used => 156 8 byte blocks
    print(f"Using max bits: {max_bits}")

    logging.info(f"Training data has {len(train_data)} samples, validation data has {len(val_data)} samples.")

    x_train = extract_feature_batches(train_data, skip_bytes, max_bits)
    x_val = extract_feature_batches(val_data, skip_bytes, max_bits)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(64, input_shape=(None, bits_per_line), recurrent_dropout=0.4))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', 'AUC', Precision(), Recall(), matthews_correlation_coefficient, f1])
    print(model.summary())
    logging.info("Fit model on training data")
    history = model.fit(
        x_train,
        train_data['label'],
        batch_size=512,
        epochs=200,
        verbose=1,
        validation_data=(x_val, val_data['label']),
        validation_batch_size=len(val_data),  # Important workaround for the custom metrics (f1 and MCC)!
        callbacks=callbacks
    )
    return model, history


def recurrent_words_attack(data, train=None, test=None):
    """
    This method is used to predict a number from a given ACRCloud fingerprint using a recurrent neural network.
    We currently assume that the fingerprint is generated from a WAV file of 2 seconds of audio and
    contains exactly one number.
    Only bytes 0,1 and 6,7 add a lot of accuracy.
    """
    skip_bytes = [2, 3, 4, 5]
    bits_per_line = 64 - (len(skip_bytes) * 8)

    max_bits = 2496

    if train is None or test is None:
        train, test = train_test_split(data, test_size=0.2, stratify=data['label'])
    x_train = extract_feature_batches(train, skip_bytes, max_bits)
    x_test = extract_feature_batches(test, skip_bytes, max_bits)

    model = Sequential()

    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(32, input_shape=(None, bits_per_line), recurrent_dropout=0.4,
                   return_sequences=True))
    model.add(LSTM(32, recurrent_dropout=0.2))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    print(model.summary())

    print("Fit model on training data")
    history = model.fit(
        x_train,
        train['label'],
        batch_size=64,
        epochs=150,
        validation_data=(x_test, test['label'])
    )
    return history
