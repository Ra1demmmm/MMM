import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math

class MixtureM(nn.Module):
    def __init__(self,
                 input_dim,
                 encoding_dim1=512,
                 encoding_dim2=512,
                 latent_dim=10,
                 c=10,
                 binary = False):
        super(MixtureM, self).__init__()

        self.binary = binary

        self.z_est_fc1 = nn.Linear(input_dim, encoding_dim1, bias=True)
        self.z_est_fc2 = nn.Linear(encoding_dim1, encoding_dim2, bias=True)
        self.z_est_fc3_mu = nn.Linear(encoding_dim2, latent_dim, bias=True)
        self.z_est_fc3_var = nn.Linear(encoding_dim2, latent_dim, bias=True)

        self.y_est_fc1 = nn.Linear(latent_dim, encoding_dim1, bias=True)
        self.y_est_fc2 = nn.Linear(encoding_dim1, encoding_dim2, bias=True)
        self.y_est_fc3 = nn.Linear(encoding_dim2, c, bias=True)

        self.decoder_fc1 = nn.Linear(latent_dim, encoding_dim2, bias=True)
        self.decoder_fc2 = nn.Linear(encoding_dim2, encoding_dim1, bias=True)
        self.decoder_fc3 = nn.Linear(encoding_dim1, input_dim, bias=True)


    def z_estimate(self, x):
        h1 = F.relu(self.z_est_fc1(x))
        h2 = F.relu(self.z_est_fc2(h1))
        z_mu = self.z_est_fc3_mu(h2)
        z_var = F.softplus(self.z_est_fc3_var(h2))

        return z_mu, z_var

    def y_estimate(self, zm):
        h1 = F.relu(self.y_est_fc1(zm))
        h2 = F.relu(self.y_est_fc2(h1))
        y_logits = self.y_est_fc3(h2)

        y_prob = torch.softmax(y_logits, dim=1)

        return y_logits, y_prob


    def reparametrize(self, mu, var):

        std = torch.sqrt(var)
        dev = mu.get_device()
        if dev == -1:
            dev = 'cpu'
        eps = torch.FloatTensor(std.size()).normal_().to(dev)
        eps = Variable(eps)

        return eps * std + mu

    def decoder(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        if self.binary:
            x_reconstruct = F.sigmoid(self.decoder_fc3(h2))
        else:
            x_reconstruct = self.decoder_fc3(h2)

        return x_reconstruct

    def forward(self, x, mode='train'):
        if mode == 'pretrain':
            zm, zv = self.z_estimate(x)
            x_rec = self.decoder(zm)

            return zm, x_rec

        elif mode == 'pretrain_y':
            zm, _ = self.z_estimate(x)
            y_logits, y_prob = self.y_estimate(zm)

            return y_logits, y_prob

        elif mode == 'train':
            zm, zv = self.z_estimate(x)
            y_logits, y_prob = self.y_estimate(zm)
            z = self.reparametrize(zm, zv)
            x_rec = self.decoder(z)

            return z, y_logits, y_prob, zm, zv, x_rec

class Estimator(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim1=512,
                 hidden_dim2=512):
        super(Estimator, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1, bias=True)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, bias=True)
        self.fc3 = nn.Linear(hidden_dim2, 1, bias=True)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        y = F.sigmoid(self.fc3(h2))

        return y


class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.student = {}

    def record(self, student_model):
        for name, param in student_model.named_parameters():
            # if param.requires_grad:
            self.student[name] = param.data.clone()

    def update(self, teacher_model):
        for name, param in teacher_model.named_parameters():
            # if param.requires_grad:
            assert name in self.student
            new_average = self.decay * param.data + (1.0 - self.decay) * self.student[name]
            param.data = new_average.clone()

