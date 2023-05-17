"""
Script to evaluate the speaker recognition attack on different fingerprint types.
"""
import argparse
import os
import random
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tikzplotlib
from joblib import dump, load
from sklearn.metrics import auc
from sklearn.model_selection import GroupKFold

import attacking.common.config as config
from attacking.attacks import acrcloud_attack, sonnleitner_attack, zapr_attack
from attacking.common.config import (DATAPATHS,
                                     SPEAKER_LIBRISPEECH_TEST_LABELENCODER)
from attacking.common.utils import extract_file_contents

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set random seed to ensure equality of CV splits
os.environ['PYTHONHASHSEED'] = str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)


sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


PLOT_THRESHOLD = 0.005
ORANGE_COLOR = (1, 0.647, 0)  # rgb color orange


def dump_results(out_root, results, fprs, tprs, roc_aucs):
    # Collect the results and output them to two csv files, one for individual scores, one for mean and std of scores.
    scores = pd.DataFrame.from_dict(results)
    scores.to_csv(os.path.join(out_root, 'individual_scores.csv'), index=False)

    dump(fprs, os.path.join(out_root, 'fprs.joblib.gz'))
    dump(tprs, os.path.join(out_root, 'tprs.joblib.gz'))
    dump(roc_aucs, os.path.join(out_root, 'roc_aucs.joblib.gz'))

    means = pd.Series(np.mean(scores), name='mean')
    stds = pd.Series(np.std(scores), name='std')
    conc = pd.concat([means, stds], axis=1)
    conc.to_csv(os.path.join(out_root, 'mean_scores.csv'))


def plot_cv_macro_curves(fprs, tprs, roc_aucs, outpath=None):
    """
    Plot the 10 macro averaged ROC curves, one for each CV run.
    """
    plt.clf()
    f = plt.figure(1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves of cross-validation')

    for i, (fpr, tpr, roc_auc) in enumerate(zip(fprs, tprs, roc_aucs)):
        tpr[0] = 0.0
        tpr[-1] = 1.0
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average ROC curve run {i}(auc = {roc_auc["macro"]:0.2f})', linewidth=1)
    plt.legend(loc="lower right")

    if outpath is not None:
        if outpath.endswith('.tex'):
            tikzplotlib.save(outpath)
        elif outpath.endswith('.pdf'):
            f.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()


def make_mean_roc_std_plot(fprs, tprs, roc_aucs, ax, cv_runs=10):
    """
    Plot the mean macro-averaged ROC curve and its standard deviation
    """
    macro_tprs = [tpr["macro"] for tpr in tprs]
    macro_fprs = [fpr["macro"] for fpr in fprs]
    macro_aucs = [x["macro"] for x in roc_aucs]

    # all needed fpr thresholds (x-values)
    all_fpr = np.unique(np.concatenate(
        [macro_fprs[i] for i in range(cv_runs)]))
    mean_tpr = np.zeros_like(all_fpr)

    tprs_interp = []
    for i in range(cv_runs):
        tpr_interp = np.interp(all_fpr, macro_fprs[i], macro_tprs[i])
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        mean_tpr += tpr_interp
        tprs_interp.append(tpr_interp)
    mean_tpr /= cv_runs

    tprs_std = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + tprs_std, 1)
    tprs_lower = np.maximum(mean_tpr - tprs_std, 0)

    std_auc = np.std(macro_aucs)
    mean_auc = np.mean(macro_aucs)

    lim_all_fpr, lim_mean_tpr, used_indices = get_limited_lists(all_fpr, mean_tpr, threshold=PLOT_THRESHOLD,
                                                                return_indices=True)
    plt.plot(lim_all_fpr, lim_mean_tpr, label=f'Mean macro-avg ROC (auc = {mean_auc:0.3f} $\pm$ {std_auc:0.3f} )',
             color='blue', linewidth=0.75)


def plot_10class_roc_curves(fprs, tprs, roc_aucs, outpath=None):
    """
    Plot the 10 roc curves that are in CV 1. This method is not used for evaluation.
    """
    fprs_0 = fprs[0]
    tprs_0 = tprs[0]
    roc_aucs_0 = roc_aucs[0]

    plt.clf()
    f = plt.figure(1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves of CV1 for each class')

    for i in range(10):
        plt.plot(fprs_0[i], tprs_0[i],
                 label=f'macro-average ROC curve run {i}(auc = {roc_aucs_0[i]:0.2f})', linewidth=1)
    plt.show()


def which_indices(x_coords, y1_coords, y2_coords=None, threshold=PLOT_THRESHOLD):
    """
    Given a sequence of (x,y) pairs, we only pick these (x,y) pairs where the values change "enough".
    If abs(x_i+1 - x_i) + abs(y_i+1 - y_i) >= threshold, we use the points i+1.
    If y2 is given as well (same shape as x and y1), we consider this as well.
    This is used to massively reduce the exported points via tikzplotlib without quality degradation.
    Returns the indices which should be used. The first and last point are guaranteed to be in the result.
    """
    if y2_coords is not None and len(y2_coords) != len(y1_coords):
        raise Exception(
            "Invalid shape. y2_coords needs to have same dimensions as x and y1.")
    used_indices = [0]
    cur_x = x_coords[0]
    cur_y1 = y1_coords[0]
    if y2_coords is not None:
        cur_y2 = y2_coords[0]
        y2_copy = y2_coords
    else:
        cur_y2 = 0
        y2_copy = np.zeros_like(y1_coords)
    for i, (x, y1, y2) in enumerate(zip(x_coords, y1_coords, y2_copy)):
        if abs(cur_x - x) + abs(cur_y1 - y1) + abs(cur_y2 - y2) >= threshold or len(x_coords) == i - 1:
            cur_x = x
            cur_y1 = y1
            cur_y2 = y2
            used_indices.append(i)
    return used_indices


def get_limited_lists(x_coords, y1_coords, y2_coords=None, threshold=PLOT_THRESHOLD, return_indices=False):
    """
    Limit the list of x and y coordinates and return the resulting coordinate pairs/triples.
    x_coords, y1_coords and y2_coords (if given) are expected to be float lists of the same length.
    The method only returns those 2d/3d points where the values in comparison to the previous point change "enough",
    i.e., more than the given threshold. This is used to reduce points of the ROC curves without visible degradation.
    """
    used_indices = which_indices(
        x_coords, y1_coords, y2_coords=y2_coords, threshold=threshold)
    lim_x = [x_coords[idx] for idx in used_indices]
    lim_y = [y1_coords[idx] for idx in used_indices]
    if y2_coords is not None:
        lim_y2 = [y2_coords[idx] for idx in used_indices]
    if return_indices:
        if y2_coords is not None:
            return lim_x, lim_y, lim_y2, used_indices
        else:
            return lim_x, lim_y, used_indices
    else:
        if y2_coords is not None:
            return lim_x, lim_y, lim_y2
        else:
            return lim_x, lim_y


def plot_extrema_mean_rocs(fprs, tprs, roc_aucs, n_classes, cv_runs=10):
    """
    Collect the 10 mean ROC curves and draw the ones with the lowest and the highest auc.
    """
    speaker_aucs = defaultdict(list)
    for run in range(cv_runs):
        for spkr in range(n_classes):
            speaker_aucs[spkr].append(roc_aucs[run][spkr])
    speaker_mean_aucs = [(spkr, np.mean(speaker_aucs[spkr]))
                         for spkr in range(n_classes)]
    speaker_mean_aucs.sort(key=lambda x: x[1])  # sort ascending by auc
    best_spkr_id, best_auc = speaker_mean_aucs[-1]
    best_auc_std = np.std(speaker_aucs[best_spkr_id])
    worst_spkr_id, worst_auc = speaker_mean_aucs[0]
    worst_auc_std = np.std(speaker_aucs[worst_spkr_id])

    # Collect all fprs and tprs of best speaker
    best_tprs = [tprs[run][best_spkr_id] for run in range(cv_runs)]
    best_fprs = [fprs[run][best_spkr_id] for run in range(cv_runs)]

    all_best_fpr = np.unique(np.concatenate(
        [best_fprs[run] for run in range(cv_runs)]))

    # now interpolate each of the 'cv_runs' curves at all this points and take the mean
    mean_best_tpr = np.zeros_like(all_best_fpr)
    tpr_list_interp = []  # store the tprs to compute standard deviation, if wanted
    for run in range(cv_runs):
        tpr_interp = np.interp(all_best_fpr, best_fprs[run], best_tprs[run])
        mean_best_tpr += tpr_interp
        tpr_list_interp.append(tpr_interp)
    mean_best_tpr /= cv_runs
    mean_best_tpr[0] = 0.0
    mean_best_tpr[-1] = 1.0
    all_best_fpr[0] = 0.0
    all_best_fpr[-1] = 1.0

    mean_best_auc = auc(all_best_fpr, mean_best_tpr)
    # best_tpr_stdev = np.std(tpr_list_interp, axis=0)
    # best_trp_upper_stdev = np.minimum(mean_best_tpr + best_tpr_stdev, 1)
    # best_trp_lower_stdev = np.maximum(mean_best_tpr - best_tpr_stdev, 0)

    # Now do the same for the worst speaker

    # Collect all fprs and tprs of worst speaker
    worst_tprs = [tprs[run][worst_spkr_id] for run in range(cv_runs)]
    worst_fprs = [fprs[run][worst_spkr_id] for run in range(cv_runs)]

    all_worst_fpr = np.unique(np.concatenate(
        [worst_fprs[run] for run in range(cv_runs)]))

    # now interpolate each of the 'cv_runs' curves at all this points and take the mean
    mean_worst_tpr = np.zeros_like(all_worst_fpr)
    tpr_list_interp = []  # store the tprs to compute standard deviation, if wanted
    for run in range(cv_runs):
        tpr_interp = np.interp(all_worst_fpr, worst_fprs[run], worst_tprs[run])
        mean_worst_tpr += tpr_interp
        tpr_list_interp.append(tpr_interp)
    mean_worst_tpr /= cv_runs
    mean_worst_tpr[0] = 0.0
    mean_worst_tpr[-1] = 1.0
    all_worst_fpr[0] = 0.0
    all_worst_fpr[-1] = 1.0

    mean_worst_auc = auc(all_worst_fpr, mean_worst_tpr)
    # worst_tpr_stdev = np.std(tpr_list_interp, axis=0)
    # worst_trp_upper_stdev = np.minimum(mean_worst_tpr + worst_tpr_stdev, 1)
    # worst_trp_lower_stdev = np.maximum(mean_worst_tpr - worst_tpr_stdev, 0)

    lim_all_best_fpr, lim_mean_best_tpr = get_limited_lists(
        all_best_fpr, mean_best_tpr, threshold=PLOT_THRESHOLD)
    plt.plot(lim_all_best_fpr, lim_mean_best_tpr, color=ORANGE_COLOR,
             label=f'Best speaker mean ROC (auc = {mean_best_auc:0.3f} $\pm$ {best_auc_std:0.3f})', linewidth=1)

    lim_all_worst_fpr, lim_mean_worst_tpr = get_limited_lists(
        all_worst_fpr, mean_worst_tpr, threshold=PLOT_THRESHOLD)
    plt.plot(lim_all_worst_fpr, lim_mean_worst_tpr, color='green',
             label=f'Worst speaker mean ROC (auc = {mean_worst_auc:0.3f} $\pm$ {worst_auc_std:0.3f})', linewidth=1)


def plot_final(fprs, tprs, roc_aucs, n_classes, outpath=None, cv_runs=10):
    """
    Plots serveral things:
    1. The mean one vs. rest ROC curve of the speaker that has the largest average AUC among all speakers.
    2. The mean one vs. rest ROC curve of the speaker that has the lowest average AUC among all speakers.
    3. The mean macro-averaged ROC curve.
    """
    plt.clf()
    plt.figure(dpi=1200)
    fig, ax = plt.subplots()

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    make_mean_roc_std_plot(fprs, tprs, roc_aucs, ax, cv_runs)
    plot_extrema_mean_rocs(fprs, tprs, roc_aucs, n_classes, cv_runs=cv_runs)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right")

    if outpath is not None:
        if outpath.endswith('.tex'):
            tikzplotlib.save(outpath)
        else:
            fig.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()


def call_worker(params):
    """
    Call the separate worker process to execute a cross validation run.
    """
    split_folder, method = params
    script_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'evaluation_speaker_worker.py')
    print(f"Starting worker for {split_folder}.")
    subprocess.call(['python', script_path,
                     '--method', method,
                     '--split_folder', split_folder])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-mt', '--method', type=str, required=True,
                        choices=['ACRCLOUD', 'SONNLEITNER',
                                 'ZAPR_ALG1', 'ZAPR_ALG2'],
                        help='Which fingerprinting method to use.')
    parser.add_argument('-out', '--outfolder', type=str, required=True)
    parser.add_argument('--plot_only', action='store_true')
    parser.add_argument('--n_processes', type=int, default=5)

    cv_runs = config.CV_RUNS
    args = parser.parse_args()
    main_method = args.method
    n_classes = 40
    if not args.plot_only:
        labelencoder = load(
            globals()[f"SPEAKER_LIBRISPEECH_TEST_LABELENCODER"])

        data = extract_file_contents(
            DATAPATHS[f"{main_method}_LIBRISPEECH_SPEAKER_RECOGNITION_TEST"], mode='speaker_librispeech')
        data['label'] = labelencoder.transform(data['label'])

        # assign each sample to a group based on the video id/chapter id
        # note: filenames follow the naming scheme ${SPEAKER_ID}_${VIDEO_ID}_${PART}_${CHUNK}
        data['group'] = data['filename'].str.extract('(..-..)', expand=True)

        # shuffle data as GroupKFold does not provide such an option
        data = data.sample(frac=1, random_state=1)

        feature_extract_params = {}
        if args.method == 'SONNLEITNER':
            feature_extract_fn = sonnleitner_attack.extract_batched_peak_bitstrings
            feature_extract_params = {'mode': 'both'}
        elif args.method == 'ACRCLOUD':
            feature_extract_fn = acrcloud_attack.extract_feature_batches
            feature_extract_params = {
                'skip_bytes': [2, 3, 4, 5], 'max_bits': 2496}
        elif args.method == 'ZAPR_ALG1':  # ZAPR_0
            feature_extract_fn = zapr_attack.extract_zapr0_features
            feature_extract_params = {'byte_indices': [3]}
        elif args.method == 'ZAPR_ALG2':  # ZAPR_WL2
            feature_extract_fn = zapr_attack.extract_zapr_wl2_features
            feature_extract_params = {'block_size': 1}

        cv_out_roots = []
        kfold = GroupKFold(n_splits=cv_runs)
        x = feature_extract_fn(data, **feature_extract_params)
        # Used for kfold.split insted of data DF (see kfold documentation)
        mock_data = np.zeros(len(data))

        # dump data and features
        store_path = args.outfolder
        dataframe_path = os.path.join(store_path, 'data.joblib.gz')
        features_path = os.path.join(store_path, 'features.joblib.gz')
        dump(data, dataframe_path)
        dump(x, features_path)

        for cv_num, (train_index, test_index) in enumerate(kfold.split(mock_data, groups=data['group'])):
            print(f"Starting with CV {cv_num}.")

            cv_out_root = os.path.join(args.outfolder, f'split_{cv_num}')
            cv_out_roots.append(cv_out_root)
            os.makedirs(cv_out_root)
            x_train, y_train = x[train_index], data['label'].iloc[train_index]
            x_test, y_test = x[test_index], data['label'].iloc[test_index]

            dump(x_train, os.path.join(cv_out_root, 'x_train.joblib.gz'))
            dump(x_test, os.path.join(cv_out_root, 'x_test.joblib.gz'))
            dump(y_train, os.path.join(cv_out_root, 'y_train.joblib.gz'))
            dump(y_test, os.path.join(cv_out_root, 'y_test.joblib.gz'))

        del data
        del x

        # Now call individual workers to process each folder
        with Pool(processes=args.n_processes) as p:
            p.map(call_worker, [(split_folder, args.method)
                  for split_folder in cv_out_roots])
    else:
        cv_out_roots = [os.path.join(
            args.outfolder, f'split_{i}') for i in range(cv_runs)]
        print("Only reading stored results.")

    # now load in individual fprs, tprs, roc_aucs, etc.
    fprs, tprs, roc_aucs = [], [], []
    results = defaultdict(list)
    for split_folder in cv_out_roots:
        fprs.append(load(os.path.join(split_folder, 'fpr.joblib.gz')))
        tprs.append(load(os.path.join(split_folder, 'tpr.joblib.gz')))
        roc_aucs.append(load(os.path.join(split_folder, 'roc_auc.joblib.gz')))
        split_results = load(os.path.join(
            split_folder, 'split_results.joblib.gz'))
        for metric_name, score in split_results.items():
            results[metric_name].append(score)
        macro_files = load(os.path.join(
            split_folder, 'macro_scores.joblib.gz'))
        for metric_name, score in macro_files.items():
            results[metric_name].append(score)
    print("Successfully loaded results from all CV splits")

    if not args.plot_only:
        print("Finished. Now dumping results and making plot.")
        dump_results(args.outfolder, results, fprs, tprs, roc_aucs)

    plot_final(fprs, tprs, roc_aucs,
               n_classes=n_classes,
               outpath=None,
               cv_runs=cv_runs)
    plot_final(fprs, tprs, roc_aucs,
               n_classes=n_classes,
               outpath=os.path.join(
                   args.outfolder, f"evaluation_plot_spkr_mean_std_{args.method}_no_bounds.png"),
               cv_runs=cv_runs)
    plot_final(fprs, tprs, roc_aucs,
               n_classes=n_classes,
               outpath=os.path.join(
                   args.outfolder, f"evaluation_plot_spkr_mean_std_{args.method}_no_bounds.pdf"),
               cv_runs=cv_runs)
    plot_final(fprs, tprs, roc_aucs,
               n_classes=n_classes,
               outpath=os.path.join(
                   args.outfolder, f"evaluation_plot_spkr_mean_std_{args.method}_no_bounds.tex"),
               cv_runs=cv_runs)
