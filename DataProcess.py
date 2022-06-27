import numpy as np
from ax.utils.tutorials.cnn_utils import load_mnist
import scipy.io as sio
from graph_filtering import graph_filtering_for_graph
from load_data import Switcher


Data_list = ["Caltech101_7", "Caltech101_20", "Citeseer", "ACM", "DBLP","IMDB","Amazon_photos","Amazon_photos_computers","AIDS"]

def dt_loader_DL(batch_size=128, dataset="Mnist") :
    if dataset == "Mnist" :
        train_loader, valid_loader, test_loader = load_mnist(batch_size=batch_size)

    return train_loader, valid_loader, test_loader



def dt_loader_ML(dataset="ACM") :

    idx = Data_list.index(dataset)
    X, gnd = Switcher[idx]()
    H, A = graph_filtering_for_graph(X, filter_order=2, dtname=dataset)
    H = np.array(H)
    A = np.array(A)
    return H, A, gnd

