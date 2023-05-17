import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf
from joblib import dump, load
from sklearn.metrics import (auc, classification_report, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import TensorBoard

from attacking.attacks import acrcloud_attack, sonnleitner_attack, zapr_attack
from attacking.common.config import SPEAKER_LIBRISPEECH_TEST_LABELENCODER

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Set random seed to ensure equality of CV splits
os.environ['PYTHONHASHSEED'] = str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-mt', '--method', type=str, required=True,
                        choices=['ACRCLOUD', 'SONNLEITNER', 'ZAPR_ALG1', 'ZAPR_ALG2'])
    parser.add_argument('--split_folder', type=str, required=True,
                        help='Output folder that already contains x_train.joblib.gz, x_test.joblib.gz, y_train.joblib.gz and '
                             'y_test.joblib.gz')

    args = parser.parse_args()

    model_params = {'n_classes': 40}
    # Now fit the models with the optimized parameters
    if args.method == 'SONNLEITNER':
        build_model_fn = sonnleitner_attack.build_speaker_recognition_model
        fit_params = {'epochs': 500, 'batch_size': 64}
    elif args.method == 'ACRCLOUD':
        build_model_fn = acrcloud_attack.build_speaker_recognition_model
        fit_params = {'epochs': 1000, 'batch_size': 64}
    elif args.method == 'ZAPR_ALG1':
        build_model_fn = zapr_attack.build_speaker_recognition_model
        fit_params = {'epochs': 600, 'batch_size': 64}
    elif args.method == 'ZAPR_ALG2':
        build_model_fn = zapr_attack.build_speaker_recognition_model
        fit_params = {'epochs': 50, 'batch_size': 256}
        model_params.update({
            'lstm_units': 80,
            'lstm_activation': 'tanh',
            'lstm_rec_do': 0.4,
            'dropout_after': 0.0,
            'learning_rate': 0.005,
            'bits_per_line': 8}
        )
    labelencoder = load(globals()[f"SPEAKER_LIBRISPEECH_TEST_LABELENCODER"])

    x_train = load(os.path.join(args.split_folder, 'x_train.joblib.gz'))
    x_test = load(os.path.join(args.split_folder, 'x_test.joblib.gz'))
    y_train = load(os.path.join(args.split_folder, 'y_train.joblib.gz'))
    y_test = load(os.path.join(args.split_folder, 'y_test.joblib.gz'))

    model = build_model_fn(**model_params)

    tensorboard_callback = TensorBoard(log_dir=os.path.join(args.split_folder, 'tb'), histogram_freq=0,
                                       write_graph=False, profile_batch=0)
    callbacks = [
        tensorboard_callback,
    ]

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              callbacks=callbacks, verbose=0, **fit_params)
    model.save(os.path.join(args.split_folder, 'final_model'))

    test_scores = model.evaluate(x_test, y_test, verbose=0)
    pred_scores = model.predict(x_test)
    y_pred = np.argmax(pred_scores, axis=-1)

    with open(os.path.join(args.split_folder, 'classification_report.log'), 'w') as f:
        f.write(classification_report(y_test, y_pred, target_names=[
                str(x) for x in labelencoder.classes_], digits=4))

    # dump the macro-average f1, precision, recall
    macro_scores = dict()
    macro_scores["f1"] = f1_score(y_test, y_pred, average='macro')
    macro_scores["precision"] = precision_score(
        y_test, y_pred, average='macro')
    macro_scores["recall"] = recall_score(y_test, y_pred, average='macro')

    split_results = {metric_name: score for metric_name,
                     score in zip(model.metrics_names, test_scores)}
    top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    top5_metric.update_state(y_test, pred_scores)
    split_results["top5_acc"] = top5_metric.result().numpy()

    # ROC computation, see https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
    n_classes = 40
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # Compute the fprs and tprs and auc per class. fpr[i] is a numpy array.
        fpr[i], tpr[i], _ = roc_curve(
            y_test_binarized[:, i], pred_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Flip roc curve if roc_auc < 0.5
        if roc_auc[i] < 0.5:
            temp_fpr = fpr[i].copy()
            fpr[i] = tpr[i].copy()
            tpr[i] = temp_fpr
            roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_binarized.ravel(), pred_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Now: Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    # This are all fpr thresholds (x-values) where the roc curve of at least one curve was evaluated
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    dump(pred_scores, os.path.join(args.split_folder, 'pred_scores.joblib.gz'))
    dump(y_test_binarized, os.path.join(
        args.split_folder, 'y_test_binarized.joblib.gz'))
    dump(fpr, os.path.join(args.split_folder, 'fpr.joblib.gz'))
    dump(tpr, os.path.join(args.split_folder, 'tpr.joblib.gz'))
    dump(roc_auc, os.path.join(args.split_folder, 'roc_auc.joblib.gz'))
    dump(split_results, os.path.join(args.split_folder, 'split_results.joblib.gz'))
    dump(macro_scores, os.path.join(args.split_folder, 'macro_scores.joblib.gz'))
