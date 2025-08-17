import os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

from libs.model import *
from libs.loss import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def feature_extration(z, y_logits, x_rec, x):
    eps = 1e-10

    y_prob = torch.softmax(y_logits, dim=1)
    class_entropy = -torch.sum(y_prob * torch.log(y_prob + eps), dim=1).unsqueeze(1)
    mse = mse_samples(x_rec, x).unsqueeze(1)
    cos_sim = cosine_similarity(x_rec, x).unsqueeze(1)

    feature = torch.concatenate([z, y_prob, class_entropy, mse, cos_sim], dim=1)

    return feature
def score_normalization(s, l, norm=True):
    if torch.is_tensor(s):
        if norm:
            ns = (s - torch.min(s)) / (torch.max(s) - torch.min(s))
        else:
            ns = s
        ns = ns.detach().numpy()
    elif isinstance(s, np.ndarray):
        if norm:
            ns = (s - np.min(s)) / (np.max(s) - np.min(s))
        else:
            ns = s

    auroc = roc_auc_score(l, ns)
    precision, recall, thresholds = precision_recall_curve(l, ns)
    auprc = auc(recall, precision)

    return ns, auroc, auprc


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    data_dir = './data/'
    model_dir = './models/'
    dataset = 'Letter'

    data = np.load(os.path.join(data_dir, dataset, 'data.npy'))
    label = np.load(os.path.join(data_dir, dataset, 'label.npy'))

    # mean-std
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean) / (data_std + 1e-10)


    valid_data = torch.tensor(data)
    valid_label = torch.tensor(label)

    mixture_model = torch.load(os.path.join(model_dir, dataset+'_mixture.pt'))
    estimator = torch.load(os.path.join(model_dir, dataset+'_estimator.pt'))

    with torch.no_grad():
        z, y_logits, y_prob, zm, zv, _ = mixture_model(valid_data.to(torch.float32))
        x_rec = mixture_model.decoder(zm)
        feature = feature_extration(zm, y_logits, x_rec, valid_data.to(torch.float32))
        predict = estimator(feature)

    est_score, auroc, auprc = score_normalization(predict, valid_label, norm=False)
    print('AUROC = {}, AUPRC = {}'.format(auroc, auprc))

