"""
Script to evaluate the speech vs music classifier. We expect that the model has already been trained on all training
data and that it was saved in 'args.outfolder/final_model'. It reads the test data, predicts on it, computes the
metrics and plots the corresponding ROC curve. Metrics are stored, as well.
"""
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef,\
    precision_score, recall_score,roc_curve, auc
import argparse

from attacking.common.config import DATAPATHS
from attacking.evaluation.evaluate_speaker_recognition import get_limited_lists

from attacking.common.metrics import f1, matthews_correlation_coefficient
from attacking.common.utils import extract_file_contents
from joblib import dump
from attacking.attacks import sonnleitner_attack, acrcloud_attack, zapr_attack
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tikzplotlib


def make_plot(fpr, tpr, roc_auc, outpath=None, title="Speech vs Music"):
    plt.clf()
    f = plt.figure(1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    fpr_lim, tpr_lim = get_limited_lists(fpr, tpr)
    plt.plot(fpr_lim, tpr_lim, label=f'ROC Speech vs Music (auc = {roc_auc:.3f})', color='blue')
    plt.legend(loc="lower right")

    if outpath is not None:
        if outpath.endswith('.tex'):
            tikzplotlib.save(outpath)
        elif outpath.endswith('.pdf'):
            f.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-mt', '--method', type=str, required=True,
                        choices=['ACRCLOUD', 'SONNLEITNER', 'ZAPR_ALG1', 'ZAPR_ALG2'],
                        help='Which fingerprinting method to use.')
    parser.add_argument('-out', '--outfolder', type=str, required=True,
                        help='Where to store the output. The already trained model needs to be stored in '
                             'the folder "<OUTFOLDER>/final_model". The grid search worker for speech vs music '
                             'creates this folder using TensorFlows model.save.')

    args = parser.parse_args()

    feature_extract_params = {}
    if args.method == 'SONNLEITNER':
        feature_extract_params = {'mode': 'frequency'}
        feature_extract_fn = sonnleitner_attack.extract_batched_peak_bitstrings
    elif args.method == 'ACRCLOUD':
        feature_extract_fn = acrcloud_attack.extract_feature_batches
        feature_extract_params = {'skip_bytes': [2, 3, 4, 5], 'max_bits': 3360}
    elif args.method == 'ZAPR_ALG1':
        feature_extract_fn = zapr_attack.extract_zapr0_features
        feature_extract_params = {'byte_indices': [2, 3]}
    elif args.method == 'ZAPR_ALG2':
        feature_extract_fn = zapr_attack.extract_zapr_wl2_features
        feature_extract_params = {'block_size': 1}

    test_data = extract_file_contents(DATAPATHS[f"{args.method}_SPEECHVSMUSIC_TEST"], mode='speech_vs_music')

    x_test = feature_extract_fn(test_data, **feature_extract_params)
    y_test = test_data['label']
    dump(x_test, os.path.join(args.outfolder, 'x_test.joblib.gz'))
    dump(y_test, os.path.join(args.outfolder, 'y_test.joblib.gz'))

    model = load_model(os.path.join(args.outfolder, 'final_model'),
                       custom_objects={'f1': f1, 'matthews_correlation_coefficient': matthews_correlation_coefficient})

    # This does not seem to work due to in bug in tensorflow. The problem lies in using the custom metrics f1 and mcc.
    # test_scores = model.evaluate(x_test, y_test, verbose=1, batch_size=len(x_test))

    pred_scores = model.predict(x_test)
    y_pred = (pred_scores > 0.5).astype("int32")
    results = dict()

    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred)
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['mcc'] = matthews_corrcoef(y_test, y_pred)

    with open(os.path.join(args.outfolder, 'classification_report.log'), 'w') as f:
        f.write(classification_report(y_test, y_pred, target_names=['Music', 'Speech'], digits=4))

    fpr, tpr, thresholds = roc_curve(y_test, pred_scores)
    roc_auc = auc(fpr, tpr)

    print("Now dumping ROC results and making curves.")

    dump(fpr, os.path.join(args.outfolder, 'fpr.joblib.gz'))
    dump(tpr, os.path.join(args.outfolder, 'tpr.joblib.gz'))
    dump(roc_auc, os.path.join(args.outfolder, 'roc_auc.joblib.gz'))
    dump(results, os.path.join(args.outfolder, 'results.joblib.gz'))

    with open(os.path.join(args.outfolder, 'results.txt'), 'w') as f:
        for metric_name, score in results.items():
            f.write(f"{metric_name}: {score}\n")

    title = f"SvM {args.method}"
    # make_plot(fpr, tpr, roc_auc, title=title)
    make_plot(fpr, tpr, roc_auc, outpath=os.path.join(args.outfolder, f"evaluation_plot_SvM_{args.method}.tex"),
              title=title)
    make_plot(fpr, tpr, roc_auc, outpath=os.path.join(args.outfolder, f"evaluation_plot_SvM_{args.method}.pdf"),
              title=title)

    print("Finished.")
