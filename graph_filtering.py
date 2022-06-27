import numpy as np
import traceback


# Graph filtering on graph data
def graph_filtering_for_graph(X, filter_order=2, dtname='ACM') :
    try :
        A_list = []

        num_view = len(X) - 1
        N = X[0].shape[0]
        I = np.eye(N)
        if "Amazon" in dtname :
            H = X[:2].copy()
            A = X[2] + I
            A_list.append(A)
            D = np.sum(A, axis=1)

            D = np.power(D, -0.5)

            D[np.isinf(D)] = 0
            D = np.diagflat(D)
            A = D.dot(A).dot(D)
            L = I - A

            for i in range(num_view) :

                for k in range(filter_order) :
                    H[i] = (I - 0.5 * L).dot(H[i])
                print("filtering No. {}!!".format(i))
        else :
            print("Begin Filtering!")
            A_ = X[1 :].copy()

            H = []
            for i in range(num_view) :
                print("Begin Filtering {}!".format(i + 1))
                H.append(X[0])
                A = A_[i] + I
                A_list.append(A)
                D = np.sum(A, axis=1)

                D = np.power(D, -0.5)
                # D_[np.isinf(D_)] = 0
                # D_ = np.diagflat(D_)
                D[np.isinf(D)] = 0
                D = np.diagflat(D)
                A = D.dot(A).dot(D)
                L = I - A

                # _, Nbrs_list = find_subgraphs(A)
                # order_list = filter_node_order(Nbrs_list=Nbrs_list, L=L, dtname=dtname, num_view=i, X=X[0])

                for k in range(filter_order) :
                    H[i] = (I - 0.5 * L).dot(H[i])

                # for k in range(filter_order):
                #     H[i] = (I - 0.5 * L).dot(H[i])
                print("filtering No. {}!!".format(i))

        return H, A_list
    except Exception :
        traceback.print_exc()


