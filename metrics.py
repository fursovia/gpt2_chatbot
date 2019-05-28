import torch
import numpy as np


def calculate_mrr(y_true, preds):
    matches = np.equal(y_true, preds).astype(np.int32)

    matches = matches[matches.sum(axis=1) > 0]

    places = matches.argmax(axis=1) + 1

    mrr = np.sum(1 / places) / preds.shape[0]

    return mrr


def calculate_precision_at_k(context, response, k=3):
    pass

