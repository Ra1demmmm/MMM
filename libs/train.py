import copy
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data.sampler import WeightedRandomSampler
from torch import optim
from sklearn.cluster import SpectralClustering
from itertools import chain
from tqdm import tqdm

from libs.loss import *
from libs.model import *
from libs.adaptsigma import *

def dataloader_init_cuda(input, output, batch_size, *extra_tensors, mode='random', label_flag=None):
    if extra_tensors:
        torch_dataset = Data.TensorDataset(input, output, *extra_tensors)
    else:
        torch_dataset = Data.TensorDataset(input, output)

    if mode == 'random':
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

    elif mode == 'balance':
        labelled = (label_flag == 1)
        unlabelled = (label_flag == 0)
        num_all = len(label_flag)
        num_la = torch.sum(labelled)
        num_unla = torch.sum(unlabelled)
        weights = torch.zeros(num_all)
        weights[labelled] = 1. / num_la
        weights[unlabelled] = 1. / num_unla

        sampler = WeightedRandomSampler(weights=weights, num_samples=num_all, replacement=True)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            sampler=sampler,
            # batch_sampler=sampler,
            drop_last=True
        )


    return loader

def feature_extration(z, y_logits, x_rec, x):
    eps = 1e-10

    y_prob = torch.softmax(y_logits, dim=1)
    class_entropy = -torch.sum(y_prob * torch.log(y_prob + eps), dim=1).unsqueeze(1)
    mse = mse_samples(x_rec, x).unsqueeze(1)
    cos_sim = cosine_similarity(x_rec, x).unsqueeze(1)

    feature = torch.concatenate([z, y_prob, class_entropy, mse, cos_sim], dim=1)

    return feature

def MMM_train(mixture_model,
              gmm_param,
              estimator_student,
              estimator_teacher,
              train_data,
              train_label,
              train_weights,
              label_flag=None,
              num_epoch=500,
              batch_size=-1,
              mode='balance',
              lr=1e-3,
              weighting=1,
              est_init_epoch=100,
              m_ratio=0.3,
              n_step=10,
              u=1.,
              max_epoch=-1,
              ema_decay1=0.99,
              ema_decay2=0.999):


    mu_c = gmm_param[0]
    sigma2_c = gmm_param[1]

    train_data = torch.tensor(train_data).to(torch.float32)
    train_label = torch.tensor(train_label).to(torch.float32)
    train_weights = torch.tensor(train_weights).to(torch.float32)
    if torch.cuda.is_available():
        mixture_model.cuda()
        estimator_student.cuda()
        estimator_teacher.cuda()
        mu_c= mu_c.cuda()
        sigma2_c = sigma2_c.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        train_weights = train_weights.cuda()

    if max_epoch == -1:
        max_epoch = num_epoch

    if batch_size <= 0:
        batch_size = int(train_data.shape[0] // 10)

    mu_c.requires_grad = True
    sigma2_c.requires_grad = True


    optimizer1 = optim.Adam(chain(mixture_model.parameters(), estimator_student.parameters()), lr=lr)
    optimizer2 = optim.Adam([mu_c, sigma2_c], lr=lr)


    loss_fn = mixture_model_loss

    un = (train_label == -1)
    data_un_pure = train_data[un, 0]
    pos_un = torch.tensor(list(range(len(train_data)))).cuda()[un]
    n_un_all = len(data_un_pure)
    m = round(n_un_all * m_ratio)
    spl_weights = torch.ones(len(train_data)).cuda()
    spl_targets = copy.deepcopy(train_label)
    spl_targets[un] = 0
    train_loader = dataloader_init_cuda(train_data, train_label, batch_size, train_weights, spl_weights, spl_targets, mode=mode, label_flag=label_flag)
    beta_dist = torch.distributions.beta.Beta(torch.tensor([0.5]), torch.tensor([0.5]))

    loss_epochs = torch.zeros(num_epoch)

    # best_ari = 0
    ema = EMA(decay=ema_decay1)
    print('Training ...')
    for epoch in tqdm(range(num_epoch)):
        if epoch >= est_init_epoch:
            # self-paced weights computing
            mixture_model.eval()
            with torch.no_grad():
                z, y_logits, y_prob, zm, zv, x_rec = mixture_model(data_un_pure)
                y_prob = torch.softmax(y_logits, dim=1)
                ce = -torch.sum(y_prob * torch.log(y_prob + 1e-10), dim=1)
                ce = (ce - torch.min(ce)) / (torch.max(ce) - torch.min(ce))
                mse = mse_samples(x_rec, data_un_pure)
                mse = (mse - torch.min(mse)) / (torch.max(mse) - torch.min(mse))
                predict = (ce + mse) / 2

            delta = min(round(u * n_un_all), m + math.floor(min(epoch - est_init_epoch, max_epoch - est_init_epoch) / (
                    (max_epoch - est_init_epoch) / n_step)) * math.floor((u * n_un_all - m) / (n_step - 1)))

            _, inds = torch.topk(predict, k=n_un_all-delta, dim=0, largest=True)
            exclude = pos_un[inds]
            spl_weights = torch.ones(len(train_data)).cuda()
            spl_weights[exclude] = 0.
            spl_targets = copy.deepcopy(train_label)
            spl_targets[pos_un] = predict
            train_loader = dataloader_init_cuda(train_data, train_label, batch_size, train_weights, spl_weights, spl_targets,
                                                mode=mode, label_flag=label_flag)

        # forward
        loss_cum = 0
        mixture_model.train()
        estimator_student.train()
        estimator_teacher.eval()
        for step, (data_batch, label_batch, weights_batch, spl_weight, targets_batch) in enumerate(train_loader):
            n_batch = data_batch.shape[0]
            n_neighbor = data_batch.shape[1]
            dimension = data_batch.shape[2]

            un = (label_batch == -1)
            data_un = data_batch[un]
            weights_un = weights_batch[un]
            n_un = data_un.shape[0]
            ol = (label_batch == 1)
            data_ol = data_batch[ol, 0]

            target_un = data_un[:, 0, :].repeat(n_neighbor, 1)
            data_un = data_un.permute(1, 0, 2).reshape(-1, dimension)
            weights_un = weights_un.permute(1, 0).reshape(-1, 1)

            target_remake = copy.deepcopy(targets_batch)
            target_remake[:n_un] = targets_batch[un]
            target_remake[n_un:] = 1.

            data_rebuild = torch.concatenate([data_un, data_ol], dim=0)
            data_combine = torch.concatenate([data_un[:n_un], data_ol], dim=0)
            shuffle_idx = torch.randperm(data_combine.shape[0])
            data_shuffle = data_combine[shuffle_idx, :].view(data_combine.size())
            target_shuffle = target_remake[shuffle_idx].view(target_remake.size())
            dev = target_shuffle.get_device()
            if dev == -1:
                dev = 'cpu'
            lambda_sample = beta_dist.sample((n_batch, 1)).squeeze(2).to(dev)
            data_mixup = lambda_sample * data_combine + (1 - lambda_sample) * data_shuffle
            lambda_sample = lambda_sample.squeeze()
            target_mixup = lambda_sample * target_remake + (1 - lambda_sample) * target_shuffle

            data_rebuild = torch.concatenate([data_rebuild, data_mixup], dim=0)
            target_remake = torch.concatenate([target_remake, target_mixup], dim=0)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # DGG Part
            z, y_logits, y_prob, zm, zv, x_rec = mixture_model(data_rebuild)

            fm = n_un * n_neighbor
            z_un, y_logits_un, y_prob_un, zm_un, zv_un, x_rec_un = z[:fm], y_logits[:fm], y_prob[:fm], zm[:fm], zv[
                                                                                                                :fm], x_rec[
                                                                                                                      :fm]
            z_ol, y_logits_ol, y_prob_ol, zm_ol, zv_ol, x_rec_ol = z[fm:], y_logits[fm:], y_prob[fm:], zm[fm:], zv[
                                                                                                                fm:], x_rec[
                                                                                                                      fm:]

            yw = y_prob_un * weights_un
            pi_ = 0
            for idx in range(n_neighbor):
                pi_ = pi_ + yw[idx * n_un:(idx + 1) * n_un, :]

            pi_ = pi_.repeat(n_neighbor, 1)


            if weighting == 1:
                loss1 = loss_fn(y_prob_un, zm_un, zv_un, x_rec_un, target_un, pi_, mu_c, sigma2_c, weights_un)
            else:
                loss1 = loss_fn(y_prob_un, zm_un, zv_un, x_rec_un, target_un, pi_, mu_c, sigma2_c)

            if torch.any(torch.isnan(loss1)):
                raise ValueError('Loss is NaN!')

            # Estimator Part
            z_re = torch.concatenate([z_un[:n_un], z_ol], dim=0)
            y_logits_re = torch.concatenate([y_logits_un[:n_un], y_logits_ol], dim=0)
            x_rec_re = torch.concatenate([x_rec_un[:n_un], x_rec_ol], dim=0)
            data_re = torch.concatenate([data_un[:n_un], data_ol, data_mixup], dim=0)

            features = feature_extration(z_re, y_logits_re, x_rec_re, data_re)
            features_ori = features[:n_batch]
            features_mixup = features[n_batch:]

            predict_ori = estimator_student(features_ori).squeeze()
            predict_mixup = estimator_teacher(features_mixup).squeeze()
            target_ori = target_remake[:n_batch]

            predict_p2 = predict_ori[shuffle_idx]
            predict_combine = lambda_sample * predict_ori + (1 - lambda_sample) * predict_p2


            loss3 = estimator_loss(predict_combine, predict_mixup, type='bce')

            if epoch < est_init_epoch:
                loss2 = estimator_loss(predict_ori.squeeze(), target_ori, type='bce')
                loss = loss1 + 0.01 * (loss2 + loss3)
            else:
                loss2 = estimator_loss_weight(predict_ori.squeeze(), target_ori, spl_weight, type='bce')
                loss = loss1 + loss2 + loss3
                ema.decay = ema_decay2


            if torch.any(torch.isnan(loss)):
                raise ValueError('Loss is NaN!')

            # backward
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            ema.record(estimator_student)
            ema.update(estimator_teacher)

            loss_cum += loss.detach().cpu()

        loss_cum = loss_cum / (step + 1)
        loss_epochs[epoch] = loss_cum


    return mixture_model, estimator_teacher, gmm_param

def MMM_pretrain(model,
                 train_data,
                 num_epoch1=100,
                 num_epoch2=20,
                 batch_size=-1,
                 c=10,
                 lr=1e-3):

    def freeze(layer):
        for param in layer.parameters():
            param.requires_grad = False

    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.to(torch.float32).cuda()

    if batch_size <= 0:
        batch_size = int(train_data.shape[0] // 10)

    train_loader = dataloader_init_cuda(train_data, train_data, batch_size, mode='random')

    opt = optim.Adam(model.parameters(), lr=lr)

    loss_fn = mse

    print('Pretraining step 1 ...')
    for epoch in tqdm(range(num_epoch1)):
        # forward
        model.train()
        for step, (train_batch, _) in enumerate(train_loader):

            opt.zero_grad()
            zm, x_rec = model(train_batch, mode='pretrain')

            loss = loss_fn(x_rec, train_batch)

            # backward
            loss.backward()
            opt.step()


    model.eval()
    left = 0
    right = batch_size
    zm_train = []
    while True:
        train_data_batch = train_data[left:right]
        if torch.cuda.is_available():
            train_data_batch = Variable(train_data_batch).to(torch.float32).cuda(non_blocking=True)
        with torch.no_grad():
            zm, _ = model(train_data_batch, mode='pretrain')
            zm_train.append(zm)

        left = right
        right = left + batch_size
        if left == len(train_data):
            break
        elif right > len(train_data):
            right = len(train_data)

    zm_train = torch.cat(zm_train, dim=0).cpu().numpy()

    print('Pre-clustering ...')
    adjacency_matrix, _, _ = build_rbf_graph(zm_train, perplexity=30.)

    spectral = SpectralClustering(n_clusters=c, affinity='precomputed', assign_labels='cluster_qr', random_state=0, n_jobs=-1)
    spectral_labels = spectral.fit_predict(adjacency_matrix)

    means_init = np.array([zm_train[spectral_labels == i].mean(axis=0) for i in range(c)])
    vars_init = np.array([zm_train[spectral_labels == i].var(axis=0) for i in range(c)])

    mu_c = torch.from_numpy(means_init).float()
    sigma2_c = torch.from_numpy(vars_init).float()


    # pretrain y_estimator
    y_ = torch.zeros(len(zm_train), c).to(torch.float32).to(train_data.device)
    for i in range(len(y_)):
        y_[i, spectral_labels[i]] = 1
    train_loader = dataloader_init_cuda(train_data, y_, batch_size, mode='random')

    freeze(model.z_est_fc1)
    freeze(model.z_est_fc2)
    freeze(model.z_est_fc3_mu)
    freeze(model.z_est_fc3_var)
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.Adam(params, lr=lr)

    print('Pretraining step 2 ...')
    for epoch in tqdm(range(num_epoch2)):
        # forward
        model.train()
        for step, (train_batch, label_batch) in enumerate(train_loader):
            opt.zero_grad()
            y_logits, y_prob = model(train_batch, mode='pretrain_y')

            loss = label_loss_ce(y_prob, label_batch)

            # backward
            loss.backward()
            opt.step()


    # unfreeze
    for p in model.parameters():
        p.requires_grad = True

    return model, mu_c, sigma2_c

