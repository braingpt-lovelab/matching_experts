import time
import argparse
import numpy as np


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f"--- {end_time - start_time} seconds ---")
    return wrapper


def scorer_acc(PPL_A_and_B, labels):
    """
    Given samples' PPL_A and PPL_B, and labels, compute accuracy.
    """
    pred_labels = np.ones(PPL_A_and_B.shape[0], dtype=np.int32)
    for row_index, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
        if ppl_A < ppl_B:
            pred_labels[row_index] = 0
        elif ppl_A > ppl_B:
            pred_labels[row_index] = 1
        else:
            pred_labels[row_index] = -1

    acc = np.sum(pred_labels == labels) / (PPL_A_and_B.shape[0])
    return acc


def scorer_sem(PPL_A_and_B, labels):
    """
    Given samples' PPL_A and PPL_B, and labels, compute standard error of the mean.
    """
    pred_labels = np.ones(PPL_A_and_B.shape[0], dtype=np.int32)
    for row_index, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
        if ppl_A < ppl_B:
            pred_labels[row_index] = 0
        elif ppl_A > ppl_B:
            pred_labels[row_index] = 1
        else:
            pred_labels[row_index] = -1

    acc = np.sum(pred_labels == labels) / (PPL_A_and_B.shape[0])
    sem = np.sqrt(acc * (1 - acc) / PPL_A_and_B.shape[0])
    return sem


def str2bool(v):
    """
    Purpose:
    --------
        Such that parser returns boolean as input to function.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

