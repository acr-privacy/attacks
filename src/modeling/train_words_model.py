from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_addons as tfa
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras

from features.acrcloud import preprocess_acrcloud_words_dataset
from features.sonnleitner import preprocess_sonnleitner_words_dataset
from features.zapr1 import preprocess_zapr1_words_dataset
from features.zapr2 import preprocess_zapr2_words_dataset


def run_words_experiment(
    model: keras.Model,
    datadir: Path,
    outdir: Path,
    method: str,
    batch_size: int = 32,
    n_epochs: int = 300,
    n_decay_epochs: int = 300,
    lr_init: float = 1e-3,
    lr_min: float = 0.0,
    weight_decay: float = 1e-4,
    label_smoothing: float = 1e-1,
):
    _, seq_len, n_bits = model.layers[0].input_shape[0]

    extract_functions = {
        "acrcloud": preprocess_acrcloud_words_dataset,
        "sonnleitner": preprocess_sonnleitner_words_dataset,
        "zapr_alg1": preprocess_zapr1_words_dataset,
        "zapr_alg2": partial(preprocess_zapr2_words_dataset, n_bits=n_bits),
    }

    meta, x, y = extract_functions[method](datadir / method, seq_len=seq_len)

    test_idx = meta.loc[:, "split"] == "testing"
    x_test = x[test_idx]
    y_test = y[test_idx]

    train_idx = meta.loc[:, "split"] == "training"
    x_train = x[train_idx]
    y_train = y[train_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(x_train, y_train))

    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr_init,
        first_decay_steps=n_decay_epochs,
        t_mul=1.0,
        m_mul=1.0,
        alpha=lr_min,
    )
    optimizer = tfa.optimizers.AdamW(
        learning_rate=schedule, weight_decay=weight_decay
    )
    loss = keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing
    )
    metrics = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    checkpoint_filepath = outdir / "checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    try:
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_val, y_val),
            callbacks=[checkpoint_callback],
        )

        pd.DataFrame(history.history).to_csv(
            outdir / "history.csv", index_label="epoch"
        )

    except KeyboardInterrupt:
        pass

    model.load_weights(checkpoint_filepath)
    model.save(outdir / "final_model.keras")
    result = model.evaluate(x_test, y_test, return_dict=True)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    result["accuracy_keras"] = result["accuracy"]
    result["accuracy"] = accuracy_score(y_true, y_pred)
    result["f1score"] = f1_score(y_true, y_pred, average="weighted")
    result["recall"] = recall_score(y_true, y_pred, average="weighted")
    result["precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0.0
    )

    pd.DataFrame(result, index=[0]).to_csv(outdir / "result.csv", index=False)
