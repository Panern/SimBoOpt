
import numpy as np
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from TrainAndEvaluate import train, evaluate
from sklearn.cluster import SpectralClustering
from Metrics import metric_all
import torch

from ax.service.managed_loop import optimize
from Model import ML_model, DL_net
from DataProcess import dt_loader_ML, dt_loader_DL





class BoParas() :

    def __init__(self, type="DL", num_BoOpt=100, random_seed = 12345, Ex_name="MvAGC") :
        '''
        Args:
            :type
                "DL" or "ML"
            :datafunc
                data and lables for training and testing

            :num_BoOpt:
                iteration number of BayesianOptimization
            :random_seed:
                random seed for reproduce result
            :Ex_name
                name for Bo and writting logs
        '''

        self.type = type
        self.optimal_paras = None
        self.re = None
        self.experiment = None
        self.num_BoOpt = num_BoOpt
        self.random_seed = random_seed
        self.Ex_name = Ex_name
        if type == "DL":
             self.train_loader, self.valid_loader, self.test_loader = dt_loader_DL()
        elif type == "ML":
            self.features, self.A, self.labels = dt_loader_ML()
        else:
            print("Unsupported Task! \n")
        

    def BoML(self, random_seed2=21,  **bopt_paras):
        '''
        Args:
            :random_seed2
                random_state for reproducing of clustering
            :bopt_paras
                :guiding
                    You can just observe the result with given paras by yourself. This can be very useful for Parameters Analysis!
                    :g_paras
                        Given a set of paras with which you want to test model
                :pbounds
                    This is the low and high bound of  various paras
        '''



        from bayes_opt import BayesianOptimization
        model = ML_model(features=self.features, adj=self.A, labels=self.labels)


        #This is just for graph clustering, you can change it depending on your projects.
        def re_ML(**paras):

            num_labels = len(np.unique(model.labels))
            S = model.train(**paras)
            C = 0.5 * (np.fabs(S) + np.fabs(S.T))

            SpC = SpectralClustering(n_clusters=num_labels, affinity='precomputed', random_state=random_seed2)
            predict_labels = SpC.fit_predict(C)

            re = metric_all.clustering_metrics(predict_labels, model.labels)
            ac, nm, ari, f1, pur = re.evaluationClusterModelFromLabel()

            return ac

        Model_op = BayesianOptimization(f=re_ML,
                                        pbounds= bopt_paras["pbounds"],
                                        # pbounds={
                                        #         'alpha' : (1e-3, 1e4),
                                        #         'gama': (-5, -1)},
                                        random_state=self.random_seed,
                                        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent

                                        )
        if bopt_paras['guiding']['guiding_bool']:
            Model_op.probe(
                    params=bopt_paras['guiding']["g_paras"],
                    lazy=True
                    )
            logger_ML = JSONLogger(path="./Logging/ML/log_{}_PA.json".format(self.Ex_name))
            Model_op.subscribe(Events.OPTIMIZATION_END, logger_ML)
            Model_op.maximize(init_points=0, n_iter=0)

        else:

            logger_ML = JSONLogger(path="./Logging/ML/log_{}_main.json".format(self.Ex_name))
            Model_op.subscribe(Events.OPTIMIZATION_STEP, logger_ML)
            Model_op.maximize(
                                init_points=2, #this is for diversifying the exploration space
                                n_iter=self.num_BoOpt
                              )

        self.re= {"{}".format("ACC"): Model_op.max.get('target')}
        self.optimal_paras = Model_op.max.get('params')


    def BoDL(self, parameters=None) :
        '''
        :return: 
            best_parameters, values, experiment
        '''
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(self.random_seed)
        #initial paras

        parameters_init = {
                        "num_epochs" : 1,
                        "lr" : 0.001,
                        "momentum" : 0.0,
                        "weight_decay" : 0.0,
                    }

        net = train(
                    net=DL_net(), train_loader=self.train_loader, \
                    parameters=parameters_init, dtype=torch.float, device=device
                    )

        evaluate(net=net, data_loader=self.valid_loader, dtype=torch.float, device=device)

        def train_evaluate(parameters) :
            net = DL_net()
            net = train(
                        net=net, train_loader=self.train_loader, \
                        parameters=parameters, dtype=dtype, device=device
                        )
            re = evaluate(
                        net=net,
                        data_loader=self.valid_loader,
                        dtype=dtype,
                        device=device,
                         )
            return re



        best_parameters, values, experiment, model = optimize(
                                                    parameters=parameters,
                                                    evaluation_function=train_evaluate,
                                                    objective_name='accuracy',
                                                    total_trials=self.num_BoOpt,
                                                    random_seed=self.random_seed,
                                                    experiment_name=self.Ex_name,
                                                                )



        self.optimal_paras = best_parameters
        self.re = values
        self.experiment = experiment

    def show(self):
        if self.type == "ML" or self.type == "DL":
            print("_____________________________________________________________")
            print("_____________Optimal result is {} __________________".format(self.re))
            print("_____________Optimal paras are {} __________________".format(self.optimal_paras))
            print("_____________________________________________________________")
        else:
            print("Unsupported task!! \n")

    def Opt(self, random_seed2=21, **params):
        if self.type == "ML":
            self.BoML(random_seed2=random_seed2, **params)
        elif self.type == "DL":
            self.BoDL(params["parameters"])
        else:
            print("Unsupported task!! \n")
        self.show()


if __name__ == "__main__":
    pass

    # T1 = BoParas(type="DL", datafunc=dt_loader_DL)
    # T1.BoDL(
    #         parameters=[
    #                     {"name" : "lr", "type" : "range", "bounds" : [1e-6, 0.4], "log_scale" : True},
    #                     {"name" : "momentum", "type" : "range", "bounds" : [0.0, 1.0]},
    #                     {"name" : "num_epochs", "type" : "range", "bounds" : [1, 50]},
    #                                 ])
    # print(T1.re, T1.optimal_paras)


    # T2 = BoParas(type='ML', datafunc=dt_loadr_ML)
    # T2.BoML()
    # print(T2.re, T2.optimal_paras)

