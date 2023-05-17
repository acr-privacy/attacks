from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from attacking.common.metrics import f1, matthews_correlation_coefficient
from attacking.features.zapr_features import extract_bytes3_4_integral, extract_byte_blocks, zapr_0_bytes
import numpy as np
import pandas as pd
import logging


def extract_zapr_wl2_features(data, block_size):
    """
    Extract the bytes of a Zapr WL2 fingerprint. The block size (in byte) states how many bytes we group together.
    We do not skip any bytes of a block since we do not know their meaning.
    """
    return extract_byte_blocks(data['content'], block_size)


def extract_zapr0_features(data, byte_indices=None):
    """
    Extract Zapr0 (SC1) features for an LSTM. The list "byte_indices" contains the indices of bytes of each 4-byte block
    that should be concatenated. Thus, byte_indices can contain values 0, 1, 2, 3.
    """
    if byte_indices is None:
        # As default, extract only the last two bytes of each block
        byte_indices = [2, 3]
    return zapr_0_bytes(data["content"], byte_indices)


def build_speechvsmusic_model(lstm_units=32, lstm_activation='tanh', lstm_rec_do=0.4, dropout_after=0.2,
                              learning_rate=0.005, bits_per_line=16):
    """
    Build the model for differentiating speech and music based on given parameters.
    The default parameters are for Zapr0 (i.e., the SC1 fingerprints) and were computed as result of a grid search.
    Note that this method is used for WL2 fingerprints, as well.
    """
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
        bits_per_line=8
):
    """
    Build the model for recognizing 1 out of `n_classes` speakers based on given parameters.
    The default parameters are for Zapr0 (i.e., the SC1 fingerprints) and were computed as result of a grid search.
    Note that this method is used for WL2 fingerprints, as well.
    """
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


def build_words_model(num_classes=10, lstm1_units=32, lstm1_rec_do=0.4, lstm2_units=0, lstm2_rec_do=0.0, lstm_activation='tanh',
                      dropout_after=0.0, learning_rate=0.005, bits_per_line=16):
    """
    Build the model for the 10 class words attack (numbers 0-9) based on given parameters.
    The default parameters are for Zapr0 (i.e., the SC1 fingerprints) and were computed as result of a grid search.
    Note that this method is used for WL2 fingerprints, as well.
    """
    use_two_lstms = (lstm2_units > 0)
    model = Sequential()

    model.add(Masking(mask_value=0, input_shape=(None, bits_per_line)))
    model.add(LSTM(lstm1_units, recurrent_dropout=lstm1_rec_do,
                   return_sequences=use_two_lstms, activation=lstm_activation))
    if use_two_lstms:
        model.add(LSTM(lstm2_units, recurrent_dropout=lstm2_rec_do, activation=lstm_activation))
    if dropout_after > 0:
        model.add(Dropout(dropout_after))

    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    return model


# ------------------------------------
# The following methods are for additional optimization and evaluation (not the main results).
# They are used to determine speech or music with an SVM and a feed-forward network, respectively.
# ------------------------------------

def build_zapr0_speechvsmusic_feedforward_model(dense1_units=500, dense_activation='relu', dense2_units=50,
                                                dropout1=0.2, dropout2=0.2, learning_rate=0.001):
    """
    Build model for Zapr0 speech vs music feed forward network. This is used for an additional evaluation.
    """
    model = Sequential()
    model.add(Dense(dense1_units, activation=dense_activation, input_shape=(1280,)))
    if dropout1 > 0:
        model.add(Dropout(dropout1))
    if dense2_units > 0:
        model.add(Dense(dense2_units, activation=dense_activation))
        if dropout2 > 0:
            model.add(Dropout(dropout2))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', 'AUC', Precision(), Recall(), matthews_correlation_coefficient, f1])
    return model


def zapr_0_speechvsmusic_feedforward_gs(train_data, val_data):
    """
    Perform grid search for Zapr0 feed-forward network for speech vs music.
    """
    byte_indices = [2, 3]

    x_train = extract_zapr0_features(train_data, byte_indices)
    x_train = x_train.reshape(x_train.shape[0], -1)  # flatten to 2D
    x_val = extract_zapr0_features(val_data, byte_indices)
    x_val = x_val.reshape(x_val.shape[0], -1)

    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = pd.concat([train_data, val_data])

    fold_indices = [-1 for _ in range(len(x_train))]
    fold_indices += [0 for _ in range(len(x_val))]

    ps = PredefinedSplit(fold_indices)

    classifier = KerasClassifier(build_fn=build_zapr0_speechvsmusic_feedforward_model, verbose=0)

    model_param_grid = {'dense1_units': [100, 250, 500, 1000, 1500, 2000], 'dense_activation': ['relu', 'tanh'],
                        'dense2_units': [0, 20, 50, 100, 500], 'dropout1': [0, 0.2, 0.4],
                        'dropout2': [0, 0.2, 0.4], 'learning_rate': [0.005, 0.001, 0.0005],
                        'batch_size': [256], 'epochs': [10, 20, 30, 40, 50]}

    grid = GridSearchCV(estimator=classifier,
                        n_jobs=4,
                        verbose=10,
                        return_train_score=True,
                        cv=ps,
                        param_grid=model_param_grid,
                        refit=False)

    grid_result = grid.fit(x_all, y_all['label'])

    dump(grid_result, 'grid_result_feedforward_sc1.joblib')
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")


def zapr_0_svm_speechsvsmusic_gs(train_data, val_data):
    """
    Perform a Grid search for the Zapr SC1 speech vs. music experiment with an SVM.
    As a result (with train/val. split): C=1, kernel=rbf.
    """
    from sklearn.model_selection import PredefinedSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    x_train = extract_bytes3_4_integral(train_data)
    x_train = scaler.fit_transform(x_train, train_data['label'])

    x_val = extract_bytes3_4_integral(val_data)
    x_val = scaler.transform(x_val)

    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = pd.concat([train_data, val_data])
    fold_indices = [-1 for _ in range(len(x_train))]
    fold_indices += [0 for _ in range(len(x_val))]

    ps = PredefinedSplit(fold_indices)

    param_grid = [
        {'C': [0.1, 1, 10], 'gamma': ['scale'], 'kernel': ['rbf']},
        {'C': [0.1, 1, 10], 'kernel': ['linear']},
    ]
    svm = SVC()

    gs = GridSearchCV(svm, param_grid=param_grid, cv=ps, n_jobs=4, verbose=10, refit=False)
    gs.fit(x_all, y_all['label'])

    print(gs.cv_results_)


def zapr_0_evaluate_speechvsmusic_svm(train_data, test_data):
    """
    Evaluate the SVM classifier on the Zapr SC1 speech vs music. The params (C=1 and rbf kernel) were found
    using a GS on the validation data.
    """
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    x_train = extract_bytes3_4_integral(train_data)
    x_test = extract_bytes3_4_integral(test_data)

    svm = SVC(C=1, gamma='scale', kernel='rbf')
    pipeline = Pipeline([('scaler', scaler), ('svm', svm)])
    logging.info("Start fitting pipeline")
    pipeline.fit(x_train, train_data['label'])
    logging.info("Now predicting pipeline")
    scores = pipeline.score(x_test, test_data['label'])
    print(f"Achieved an accuracy of {scores} on the test data")

    preds = pipeline.predict(x_test)
    results = dict()

    results['accuracy'] = accuracy_score(test_data['label'], preds)
    results['f1'] = f1_score(test_data['label'], preds)
    results['recall'] = recall_score(test_data['label'], preds)
    results['precision'] = precision_score(test_data['label'], preds)
    results['mcc'] = matthews_corrcoef(test_data['label'], preds)
    print(f"Results: {results}")


def zapr_0_evaluate_speechvsmusic_feedforward(train_data, val_data):
    """
    Evaluate the feed-forward NN on Zapr0 (SC1) speech vs music fingerprints.
    Parameters were optimized with a grid search.
    """
    byte_indices = [2, 3]
    x_train = extract_zapr0_features(train_data, byte_indices)
    x_train = x_train.reshape(x_train.shape[0], -1)  # flatten to 2D
    x_val = extract_zapr0_features(val_data, byte_indices)
    x_val = x_val.reshape(x_val.shape[0], -1)

    model = Sequential()
    model.add(Dense(1500, activation='relu', input_shape=(1280,)))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', 'AUC', Precision(), Recall(), matthews_correlation_coefficient, f1])

    history = model.fit(x_train, train_data['label'], epochs=50, batch_size=256,
                        validation_batch_size=len(x_val), validation_data=(x_val, val_data['label']))
    scores = model.evaluate(x_val, val_data['label'])

    print(scores)
    return history


# ------------------------------------
# The following methods are only for experimenting with different settings/ideas and not relevant for evaluation.
# ------------------------------------

def words_attack(data):
    train, test = train_test_split(data, test_size=0.2, stratify=data['label'])
    block_size = 2
    bits_per_block = block_size * 8

    x_train = extract_byte_blocks(train['content'], block_size)
    x_test = extract_byte_blocks(test['content'], block_size)

    model = Sequential()

    model.add(Masking(mask_value=0, input_shape=(None, bits_per_block)))
    model.add(LSTM(120, return_sequences=True))
    model.add(LSTM(32))

    model.add(Dense(10, activation='softmax'))
    opt = Adam(learning_rate=0.0005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])
    print(model.summary())

    print("Fit model on training data")
    history = model.fit(
        x_train,
        train['label'],
        batch_size=256,
        epochs=100,
        validation_data=(x_test, test['label']),
        verbose=1
    )
    return history, model


def speech_vs_music_attack(train_data, val_data):
    block_size = 1
    bits_per_block = block_size * 8

    x_train = extract_byte_blocks(train_data['content'], block_size)
    x_val = extract_byte_blocks(val_data['content'], block_size)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_block)))
    model.add(LSTM(64, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    print(model.summary())

    print("Fit model on training data")
    history = model.fit(
        x_train,
        train_data['label'],
        batch_size=512,
        epochs=120,
        validation_data=(x_val, val_data['label']),
    )
    return model, history


def speaker_recognition_attack(data, train_data=None, val_data=None):
    """
    Recognize the exact speaker from a set of possible speakers.
    """
    block_size = 1
    bits_per_block = block_size * 8
    if train_data is None or val_data is None:
        train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=1)

    logging.info(f"Training data has {len(train_data)} samples, test data has {len(val_data)} samples.")

    x_train = extract_byte_blocks(train_data['content'], block_size)
    x_val = extract_byte_blocks(val_data['content'], block_size)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_block)))
    model.add(LSTM(80, recurrent_dropout=0.4))

    model.add(Dense(40, activation='softmax'))
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])

    logging.info("Fit model on training data")
    history = model.fit(
        x_train,
        train_data['label'],
        batch_size=256,
        epochs=50,
        verbose=1,
        validation_data=(x_val, val_data['label'])
    )
    return model, history


def zapr_0_words_attack(data):
    train, test = train_test_split(data, test_size=0.2, stratify=data['label'])
    byte_indices = [2, 3]
    bits_per_block = 8 * len(byte_indices)

    x_train = extract_zapr0_features(train, byte_indices=byte_indices)
    x_test = extract_zapr0_features(test, byte_indices=byte_indices)
    model = Sequential()

    model.add(Masking(mask_value=0, input_shape=(None, bits_per_block)))
    model.add(LSTM(32, recurrent_dropout=0.4))
    model.add(Dense(10, activation='softmax'))
    opt = Adam(learning_rate=0.005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])
    print(model.summary())

    print("Fit model on training data")
    history = model.fit(
        x_train,
        train['label'],
        batch_size=256,
        epochs=400,
        validation_data=(x_test, test['label']),
        verbose=1
    )
    return history, model


def zapr_0_speech_vs_music_attack(train_data, val_data):
    byte_indices = [2, 3]
    bits_per_block = 8 * len(byte_indices)

    x_train = extract_zapr0_features(train_data, byte_indices=byte_indices)
    x_val = extract_zapr0_features(val_data, byte_indices=byte_indices)
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_block)))
    model.add(LSTM(64, recurrent_dropout=0.2))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
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


def zapr_0_speaker_recognition_attack(data, train_data=None, val_data=None):
    """
    Recognize the exact speaker from a set of possible speakers.
    """
    byte_indices = [3]
    bits_per_block = 8 * len(byte_indices)
    if train_data is None or val_data is None:
        train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=1)

    logging.info(f"Training data has {len(train_data)} samples, test data has {len(val_data)} samples.")

    x_train = extract_zapr0_features(train_data, byte_indices=byte_indices)
    x_val = extract_zapr0_features(val_data, byte_indices=byte_indices)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, bits_per_block)))
    model.add(LSTM(32, recurrent_dropout=0.4))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='softmax'))
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])

    print(model.summary())
    logging.info("Fit model on training data")
    history = model.fit(
        x_train,
        train_data['label'],
        batch_size=64,
        epochs=600,
        verbose=1,
        validation_data=(x_val, val_data['label'])
    )
    return model, history
