import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sklearn.metrics import classification_report, roc_curve, auc
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import label_binarize
import argparse

from attacking.common.config import DATAPATHS, WORDS_LABELENCODER
from attacking.common.utils import extract_file_contents
from attacking.evaluation.evaluate_speaker_recognition import get_limited_lists
from joblib import load, dump
from attacking.attacks import acrcloud_attack, sonnleitner_attack, zapr_attack
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


def make_plot(fpr, tpr, roc_auc, outpath=None):
    """
    Plot the micro and macro averaged ROC-curves.
    """
    plt.clf()
    f = plt.figure(1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Word Identification')

    # temporary bug fix: ensure that tpr of 0 is 0 (so the plot starts in the lower left corner)
    tpr["macro"][0] = 0.0
    limited_macro_fpr, limited_macro_tpr = get_limited_lists(fpr["macro"], tpr["macro"])
    plt.plot(limited_macro_fpr, limited_macro_tpr, color='blue',
             label=f'macro-avg ROC (auc = {roc_auc["macro"]:0.3f})', linewidth=1)

    limited_micro_fpr, limited_micro_tpr = get_limited_lists(fpr["micro"], tpr["micro"])
    plt.plot(limited_micro_fpr, limited_micro_tpr, color='green',
             label=f'micro-avg ROC (auc = {roc_auc["micro"]:0.3f})', linewidth=1)
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
                        help='Select the fingerprinting method to use.')
    parser.add_argument('-out', '--outfolder', type=str, required=True, help='Where to store the output data.')
    parser.add_argument('--plot_only', action='store_true', help='If set, only load already computed metrics and plot them.')

    args = parser.parse_args()
    main_method = args.method
    if not args.plot_only:
        train_data = extract_file_contents(DATAPATHS[f'{main_method}_WORDS_TRAINING'], mode='number')
        test_data = extract_file_contents(DATAPATHS[f'{main_method}_WORDS_TESTING'], mode='number')

        labelencoder = load(WORDS_LABELENCODER)
        train_data['label'] = labelencoder.transform(train_data['label'])
        test_data['label'] = labelencoder.transform(test_data['label'])

        feature_extract_params = {}
        model_params = {'num_classes': 35}
        if args.method.startswith('SONNLEITNER'):
            build_model_fn = sonnleitner_attack.build_words_model
            feature_extract_fn = sonnleitner_attack.extract_batched_peak_bitstrings
            feature_extract_params = {'mode': 'both'}
            fit_params = {'epochs': 120, 'batch_size': 256}
        elif args.method.startswith('ACRCLOUD'):
            build_model_fn = acrcloud_attack.build_words_model
            feature_extract_fn = acrcloud_attack.extract_feature_batches
            feature_extract_params = {'max_bits': 1776, 'skip_bytes': [4, 5]}
            fit_params = {'epochs': 100, 'batch_size': 64}
        elif args.method.startswith('ZAPR_ALG1'):
            build_model_fn = zapr_attack.build_words_model
            feature_extract_fn = zapr_attack.extract_zapr0_features
            feature_extract_params = {'byte_indices': [2, 3]}
            fit_params = {'epochs': 400, 'batch_size': 256}
        elif args.method == 'ZAPR_ALG2':
            build_model_fn = zapr_attack.build_words_model
            feature_extract_fn = zapr_attack.extract_zapr_wl2_features
            feature_extract_params = {'block_size': 2}
            model_params.update({'lstm1_units': 120, 'lstm1_rec_do': 0.0, 'lstm2_units': 32, 'lstm2_rec_do': 0.0,
                                 'lstm_activation': 'tanh', 'dropout_after': 0.0, 'learning_rate': 0.0005,
                                 'bits_per_line': 16})
            fit_params = {'epochs': 100, 'batch_size': 256}
        else:
            raise Exception()
        x_train = feature_extract_fn(train_data, **feature_extract_params)
        x_test = feature_extract_fn(test_data, **feature_extract_params)

        os.makedirs(args.outfolder)

        dump(x_train, os.path.join(args.outfolder, 'x_train.joblib.gz'))
        dump(x_test, os.path.join(args.outfolder, 'x_test.joblib.gz'))
        dump(train_data['label'], os.path.join(args.outfolder, 'y_train.joblib.gz'))
        dump(test_data['label'], os.path.join(args.outfolder, 'y_test.joblib.gz'))
    
        model = build_model_fn(**model_params)

        tensorboard_callback = TensorBoard(log_dir=os.path.join(args.outfolder, 'tb'), histogram_freq=0,
                                           write_graph=False, profile_batch=0)
        callbacks = [
            tensorboard_callback,
        ]

        model.fit(x_train, train_data['label'], validation_data=(x_test, test_data['label']),
                  callbacks=callbacks, verbose=0, **fit_params)
        model.save(os.path.join(args.outfolder, 'final_model'))

        test_scores = model.evaluate(x_test, test_data['label'], batch_size=len(x_test), verbose=0)

        y_pred = np.argmax(model.predict(x_test), axis=-1)
        with open(os.path.join(args.outfolder, 'classification_report.log'), 'w') as f:
            f.write(classification_report(test_data['label'], y_pred, target_names=[str(x) for x in labelencoder.classes_], digits=4))

        results = {metric_name: score for metric_name, score in zip(model.metrics_names, test_scores)}
        # ROC computation, see https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
        n_classes = model_params['num_classes']
        y_test_binarized = label_binarize(test_data['label'], classes=range(n_classes))
        pred_scores = model.predict(x_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            # Compute the fprs and tprs and auc per class. fpr[i] is a numpy array.
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], pred_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), pred_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Now: Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))  # This are all fpr thresholds (x-values) where the roc curve of at least one curve was evaluated
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        print("Now dumping ROC results and making curves.")

        dump(fpr, os.path.join(args.outfolder, 'fpr.joblib.gz'))
        dump(tpr, os.path.join(args.outfolder, 'tpr.joblib.gz'))
        dump(roc_auc, os.path.join(args.outfolder, 'roc_auc.joblib.gz'))
        dump(results, os.path.join(args.outfolder, 'results.joblib.gz'))
    else:
        fpr = load(os.path.join(args.outfolder, 'fpr.joblib.gz'))
        tpr = load(os.path.join(args.outfolder, 'tpr.joblib.gz'))
        roc_auc = load(os.path.join(args.outfolder, 'roc_auc.joblib.gz'))

    make_plot(fpr, tpr, roc_auc, outpath=os.path.join(args.outfolder, f"evaluation_plot_words_{args.method}.tex"))
    make_plot(fpr, tpr, roc_auc, outpath=os.path.join(args.outfolder, f"evaluation_plot_words_{args.method}.pdf"))

    print("Finished.")
