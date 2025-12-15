import os
import warnings
warnings.filterwarnings("ignore")

import json
import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.neighbors._unsupervised import NearestNeighbors

from libs.model import *
from libs.train import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def find_m_neighbors(data, m):
    eps = 1e-10
    nbrs = NearestNeighbors(n_neighbors=m + 1, algorithm='auto', metric='euclidean').fit(data)
    distances, indices = nbrs.kneighbors(data)
    neighbors_data = np.zeros((data.shape[0], m + 1, data.shape[1]))

    for i in range(data.shape[0]):
        neighbors_data[i] = data[indices[i]]

    weights = np.exp(-distances**2)
    weights[:, 0] = 0.5
    remaining_values = weights[:, 1:]
    row_sums = remaining_values.sum(axis=1, keepdims=True)
    weights[:, 1:] = 0.5 * (remaining_values / (row_sums+eps))

    return neighbors_data, weights

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
    dataset = 'Letter'
    seed = 0

    data = np.load(os.path.join(data_dir, dataset, 'data.npy'))
    label = np.load(os.path.join(data_dir, dataset, 'label.npy'))
    train_label = np.load(os.path.join(data_dir, dataset, 'train_label.npy'))
    label_flag = train_label
    train_label = np.where(train_label==0, -1, 1) # -1: unlabeled data; 1:labeled anomalies

    # mean-std
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean) / (data_std + 1e-10)


    train_data = torch.tensor(data)
    valid_data = torch.tensor(data)
    train_label = torch.tensor(train_label)
    label_flag = torch.tensor(label_flag)
    valid_label = torch.tensor(label)

    with open("./data/dataset_args.json", 'r', encoding='utf-8') as f:
        args = json.load(f)

    arg = args[dataset]
    [dim, h1, h2, h3, c, m, weighting, pre_epoch1, pre_epoch2, est_init_epoch, num_epoch, batch_size, train_lr] = arg

    setup_seed(seed)

    mixture_model = MixtureM(dim, h1, h2, h3, c, binary=False)
    f_dim = h3 + c + 3
    estimator_student = Estimator(f_dim, f_dim * 2, f_dim * 2)
    estimator_teacher = Estimator(f_dim, f_dim * 2, f_dim * 2)

    model, mu_c, sigma2_c = MMM_pretrain(mixture_model,
                                         train_data,
                                         num_epoch1=pre_epoch1,
                                         num_epoch2=pre_epoch2,
                                         batch_size=batch_size,
                                         c=c)

    gmm_param = [mu_c, sigma2_c]
    neighbors_data, neighbors_weights = find_m_neighbors(train_data.numpy(), m)

    mixture_model, estimator, gmm_param = MMM_train(mixture_model,
                                                    gmm_param,
                                                    estimator_student,
                                                    estimator_teacher,
                                                    neighbors_data,
                                                    train_label,
                                                    neighbors_weights,
                                                    label_flag=label_flag,
                                                    num_epoch=num_epoch,
                                                    batch_size=batch_size,
                                                    mode='balance',
                                                    lr=train_lr,
                                                    weighting=weighting,
                                                    est_init_epoch=est_init_epoch
                                                    )


    mixture_model.cpu()
    mixture_model.eval()
    estimator.cpu()
    estimator.eval()


    with torch.no_grad():
        z, y_logits, y_prob, zm, zv, _ = mixture_model(valid_data.to(torch.float32))
        x_rec = mixture_model.decoder(zm)
        feature = feature_extration(zm, y_logits, x_rec, valid_data.to(torch.float32))
        predict = estimator(feature)

    est_score, auroc, auprc = score_normalization(predict, valid_label, norm=False)
    print('AUROC = {}, AUPRC = {}'.format(auroc, auprc))

