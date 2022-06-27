import os
import pickle as pkl
import sys
import warnings
from preprocess import load_dataset
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scanpy as sc

warnings.filterwarnings("ignore")


def Amazon_photos() :
    X = []
    Amazon = load_dataset("Data/npz/amazon_electronics_photo.npz")
    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A
    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()

    Attr = np.array(Attr)
    X.append(Attr)
    X.append(Attr.dot(Attr.T))
    X.append(np.array(Adj))
    return X, Gnd


def Amazon_photos_computers() :
    X = []
    Amazon = load_dataset("Data/npz/amazon_electronics_computers.npz")
    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A
    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()
    Attr = np.array(Attr)
    X.append(Attr)
    X.append(Attr.dot(Attr.T))
    X.append(np.array(Adj))
    return X, Gnd


def mine_pubmed() :
    names = ["ty", "ally"]
    pubmed_y = []
    for name in names :
        with open("./Data/pubmed/ind.pubmed.tx", 'rb') as f :
            if sys.version_info > (3, 0) :
                data1 = pkl.load(f, encoding='latin1')
        with open("./Data/pubmed/ind.pubmed.allx", 'rb') as f :
            if sys.version_info > (3, 0) :
                data2 = pkl.load(f, encoding='latin1')
        pubmed_x = sp.vstack((data1, data2))
        with open("./Data/pubmed/ind.pubmed.{}".format(name), 'rb') as f :
            if sys.version_info > (3, 0) :
                data = pkl.load(f, encoding='latin1')
            pubmed_y = np.zeros(19717)
            if name == "ty" :
                for i in range(1000) :
                    pubmed_y[i] = np.argmax(data[i])
                # print(pubmed_y.shape)
            if name == "ally" :
                for i in range(1000, 19717) :
                    pubmed_y[i] = np.argmax(data[i - 1000])
                # print(pubmed_y.shape)

    # pubmed_x = np.vstack((pkl.load(open("ind.pubmed.tx", "rb")), pkl.load(open("ind.pubmed.allx", "rb"))))
    # print(pubmed_x[1])
    # A = np.zeros((len()))
    with open("./Data/pubmed/ind.pubmed.graph", 'rb') as f2 :
        if sys.version_info > (3, 0) :
            data = pkl.load(f2)
            # print(len(data))
            # print(data[1])
            A = np.zeros((len(data), len(data)))
            for j in range(len(data)) :
                # A[j][j] = 1
                for k in data[j] :
                    A[j][k] = 1
            # print(data)
    pubmed_x = pubmed_x.toarray()
    X = []
    X.append(pubmed_x)
    X.append(pubmed_x.dot(pubmed_x.T))
    X.append(A)
    return X, pubmed_y


def Caltech101_7() :
    # data = Mp.loadmat('{}.mat'.format("./Data/mat/Caltech101-7"))
    mdata1 = sio.loadmat('./Data/mat/C_1_3.mat')
    mdata2 = sio.loadmat('./Data/mat/C_4_6.mat')
    mLabels = sio.loadmat('./Data/mat/C_label.mat')

    X = []
    # print(mdata1['data1'][0][0])
    X.append(np.array(mdata1['data1'][0][0]))
    X.append(np.array(mdata1['data2'][0][0]))
    X.append(np.array(mdata1['data3'][0][0]))

    X.append(np.array(mdata2['data4'][0][0]))
    X.append(np.array(mdata2['data5'][0][0]))
    X.append(np.array(mdata2['data6'][0][0]))

    gnd = np.squeeze(mLabels['labels'])


    return X, gnd


def Caltech101_20() :
    # data = Mp.loadmat('{}.mat'.format("./Data/mat/Caltech101-7"))
    mdata1 = sio.loadmat('./Data/mat/C_1_3_20.mat')
    mdata2 = sio.loadmat('./Data/mat/C_4_6_20.mat')
    mLabels = sio.loadmat('./Data/mat/C_label_20.mat')

    # print(mdata1.keys())
    # print(mdata2.keys())
    # print(mdata1['data1'].shape)
    X = []
    # print(mdata1['data1'][0][0])
    X.append(np.array(mdata1['data1'][0][0]))
    X.append(np.array(mdata1['data1'][1][0]))
    X.append(np.array(mdata1['data1'][2][0]))

    X.append(np.array(mdata2['data1'][0][0]))
    X.append(np.array(mdata2['data1'][1][0]))
    X.append(np.array(mdata2['data1'][2][0]))

    # for x in X:
    #     print(x.shape)
    gnd = np.squeeze(mLabels['labels'])

    # print(len(gnd))

    return X, gnd


def Nus() :
    data = sio.loadmat('./NUSWIDEOBJ.mat')
    # print(data.keys())
    # print(data['X'][0][1].shape)
    X = []
    for i in range(5) :
        X.append(data['X'][0][i])
    # print(data['Y'].shape)
    gnd = np.squeeze(data["Y"])
    return X, gnd


def Citeseer() :
    citation = sc.read("./Data/mtx/citeseer_cites.mtx")

    citation = sp.csr_matrix(citation.X).A

    citation[0][0] = 1
    # print(citation.shape)
    content = sc.read("./Data/mtx/citeseer_content.mtx")

    content = sp.csr_matrix(content.X).A
    # print(content.shape)

    labels = np.loadtxt("./Data/mtx/citeseer_act.txt", delimiter='\n')
    # print(len(labels))

    X = []
    X.append(content)
    X.append(citation)
    return X, labels



def Acm(dataname='ACM') :
    if dataname == "ACM" :
        # Load data
        dataset = "./Data/mat/" + 'ACM3025'
        data = sio.loadmat('{}.mat'.format(dataset))
        if (dataset == 'large_cora') :
            X = data['X']
            A = data['G']
            gnd = data['labels']
            gnd = gnd[0, :]
        else :
            X = data['feature']
            A = data['PAP']
            B = data['PLP']
            # C = data['PMP']
            # D = data['PTP']
    if sp.issparse(X) :
        X = X.todense()
    X_ = []
    A = np.array(A)
    B = np.array(B)
    X_.append(np.array(X))

    # for i in range(A.shape[0]):
    #     if A[i][i] == 1:
    #         A[i][i] = 0
    #     if B[i][i] == 1:
    #         B[i][i] = 0

    X_.append(A)
    X_.append(B)

    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X_, gnd


def Dblp() :
    ## Load data
    dataset = "./Data/mat/" + 'DBLP4057_GAT_with_idx'
    data = sio.loadmat('{}.mat'.format(dataset))
    if (dataset == 'large_cora') :
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else :
        X = data['features']
        A = data['net_APTPA']
        B = data['net_APCPA']
        C = data['net_APA']
        # D = data['PTP']—

    if sp.issparse(X) :
        X = X.todense()
    X_ = []
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # for i in range(A.shape[0]):
    #     if A[i][i] == 1:
    #         A[i][i] = 0
    #     if B[i][i] == 1:
    #         B[i][i] = 0
    #     if C[i][i] == 1 :
    #         C[i][i] = 0
    X_.append(np.array(X))
    X_.append(A)
    X_.append(B)
    X_.append(C)
    # av.append(C)
    # av.append(D)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X_, gnd


def Imdb() :
    # Load data
    dataset = "./Data/mat/" + 'imdb5k'
    data = sio.loadmat('{}.mat'.format(dataset))
    if (dataset == 'large_cora') :
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else :
        X = data['feature']
        A = data['MAM']
        B = data['MDM']
        # C = data['PMP']
        # D = data['PTP']
    if sp.issparse(X) :
        X = X.todense()
    X_ = []
    X_.append(np.array(X))
    A = np.array(A)
    B = np.array(B)

    X_.append(A)
    X_.append(B)
    # av.append(C)
    # av.append(D)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X_, gnd


def AIDS() :
    X_ = []
    Node_idx = np.load("./Data/npy/AIDS_node_idx.npy").tolist()
    X = np.load("./Data/npy/AIDS_attributes.npy")
    gnd = np.load("./Data/npy/AIDS_labels.npy")
    num_labels = len(np.unique(gnd))
    print(num_labels)
    X_.append(X)
    N = gnd.shape[0]
    for i in range(3) :
        print(i)
        Edgs = np.load("./Data/npy/AIDS_A_{}.npy".format(i))
        I = np.eye(N)
        A = np.zeros((N, N))
        for x in Edgs:
            A[Node_idx.index(x[0])][Node_idx.index(x[1])] == 1

        A = A + A.T
        X_.append(A)

    return X_, gnd

    pass


Switcher = {
        0 : Caltech101_7,
        1 : Caltech101_20,
        2 : Citeseer,
        3 : Acm,
        4 : Dblp,
        5 : Imdb,
        6 : Amazon_photos,
        7 : Amazon_photos_computers,
        8 : AIDS
        }

if __name__ == "__main__" :
    _, gnd = Citeseer()
    print(len(np.unique(gnd)))
    pass
