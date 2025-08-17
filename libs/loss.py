import torch
import math

def mse(recon_x, x):
    loss = torch.sum(torch.square(recon_x - x), dim=1)
    loss = torch.sum(loss) / x.size(0)

    return loss

def label_loss_ce(y_predict, y_true):
    eps = 1e-10

    return -torch.sum(y_true * torch.log(y_predict + eps) + (1 - y_true) * torch.log(1 - y_predict + eps)) / y_predict.shape[0]

def mse_samples(recon_x, x):
    losses = torch.sum(torch.square(recon_x - x), dim=1)

    return losses

def cosine_similarity(recon_x, x):
    losses = torch.sum(recon_x * x, dim=1) / (torch.sqrt(torch.sum(torch.square(recon_x), dim=1)) * torch.sqrt(torch.sum(torch.square(x), dim=1)))

    return losses

def mixture_model_loss(y_prob, zm, zv, x_rec, x, pi_, mu_c, sigma2_c, weight=None):
    eps = 1e-10
    n = zm.shape[0]
    k = pi_.shape[1]
    dev = zm.get_device()
    if dev == -1:
        dev = 'cpu'

    loss_label = torch.sum(y_prob * torch.log(y_prob / (pi_+eps) + eps), dim=1)

    loss_classes = torch.empty((n, k)).to(dev)
    for i in range(k):

        loss_class = -0.5 * torch.sum(
            torch.log(zv / (sigma2_c[i] + eps)) - (zv / (sigma2_c[i] + eps)) - torch.square(zm - mu_c[i]) / (
                        sigma2_c[i] + eps) + 1, dim=1)


        loss_classes[:,i] = loss_class * y_prob[:,i]

    loss_recon = torch.sum(torch.square(x_rec - x), dim=1)

    if weight is not None:
        weight = weight * 2
        loss_label = weight.squeeze() * loss_label
        loss_classes = weight * loss_classes
        loss_recon = weight.squeeze() * loss_recon

    loss = (torch.sum(loss_label) + torch.sum(loss_classes) + torch.sum(loss_recon)) / n

    return loss


def estimator_loss(predict, target, type='bce'):
    if type == 'bce':
        loss = torch.nn.functional.binary_cross_entropy(predict, target, reduce=None, reduction='mean')

    return loss

def estimator_loss_weight(predict, target, weight, type='bce', mode='average'):
    if type == 'bce':
        if mode == 'average':
            loss = torch.nn.functional.binary_cross_entropy(predict, target, weight, reduce=None, reduction='mean')
        elif mode == 'balance':
            inds = (weight == 1)
            valid_target = target[inds]
            num_un = torch.sum(valid_target == 0)
            num_ol = torch.sum(valid_target == 1)
            balance_weight = torch.zeros(len(weight), dtype=torch.float32, device=weight.device)
            mask1 = (target == 0) * (weight == 1)
            balance_weight[mask1] = len(weight) / (2 * num_un)
            mask2 = (target == 1) * (weight == 1)
            balance_weight[mask2] = len(weight) / (2 * num_ol)
            loss = torch.nn.functional.binary_cross_entropy(predict, target, balance_weight, reduce=None, reduction='mean')

    return loss