import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

'''
    We surpport ML and DL task, especially, traditional graph clustering and deep learning-based methods.
    You can create your task or model.
'''

#for example MAGC

class ML_model() :
    '''
    Multi-view attributed graph clustering
    '''

    def __init__(self, features, adj, labels, task="graph_clustering"):
        self.task = task
        if self.task == "graph_clustering":
            self.train_x = features
            self.A = adj
            self.labels = labels

    '''
        this is for updating, or rather, optimization for new learnable matrix
    '''
    def train(self, **params):


        I = np.eye(self.train_x[0].shape[0])
        A = self.A[0] + I
        D = np.sum(A, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        A = D.dot(A).dot(D)
        # kk =len( np.unique(gnd))
        Ls = D - A

        num_view = len(self.train_x)
        nada = [1 / num_view for i in range(num_view)]


        for i in range(3) :
            XtX_bar = 0
            for j in range(num_view) :
                XtX_bar = XtX_bar + nada[j] * self.train_x[j].dot(self.train_x[j].T)
            tmp = np.linalg.inv(params["alpha"] * I + XtX_bar)
            S = tmp.dot(params["alpha"] * Ls + XtX_bar)

            for j in range(num_view) :
                nada[j] = (-((np.linalg.norm(self.train_x[j].T - (self.train_x[j].T).dot(S))) ** 2 + params["alpha"] * (
                        np.linalg.norm(S - Ls)) ** 2) / params['gama']) ** (1 / (params['gama'] - 1))


        return S




#For example, CNN
class DL_net(nn.Module) :
    """
    Convolutional Neural Network.
    """

    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(8 * 8 * 20, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x) :
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 3)
        x = x.view(-1, 8 * 8 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
