from sklearn.datasets import load_diabetes
import pickle
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os
import torch


from synthcity.metrics.eval_performance import (
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from synthcity.utils import reproducibility
from synthcity.plugins import Plugins
import synthcity.logger as log
from synthcity.plugins.core.dataloader import GenericDataLoader


reproducibility.clear_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Plugins(categories=["generic"]).list()

assert device.type == 'cuda'

from DGE_data import get_real_and_synthetic
from DGE_utils import get_folder_names
from DGE_experiments import model_evaluation_experiment, predictive_experiment


# let's restrict ourselves to classification datasets
datasets = ['covid' ] 
#['moons', 'circles','cal_housing', 'adult', 'diabetes', 'breast_cancer',  'seer', 'cutract' ] 
model_name = 'ctgan_deep'  # synthetic data model
    
p_train = 0.8  # proportion of training data for generative model. Default values if None
n_models = 20  # number of models in ensemble
#max_n = 5000 # maximum number of data points to use for training generative model.
#nsyn = 10000  # number of synthetic data points per synthetic dataset. Defaults to same as generative training size if None

load = True  # results
load_syn = True  # data
save = True  # save results and data

verbose = False


for nsyn in [2000, 5000]:
    for max_n in [2000, 5000, 10000]:#, 5000, 10000]:
        if max_n>nsyn:
                continue
        for dataset in datasets:#datasets:
            for model_name in ['ctgan_deep', 'ctgan', 'ctgan_shallow']:
                print('Dataset:', dataset)

                workspace_folder, results_folder = get_folder_names(
                dataset, model_name, max_n=max_n, nsyn=nsyn)

                X_gt, X_syns = get_real_and_synthetic(dataset=dataset,
                                                    p_train=p_train,
                                                    n_models=n_models,
                                                    model_name=model_name,
                                                    load_syn=load_syn,
                                                    verbose=verbose,
                                                    max_n=max_n)

                
                y_preds, scores = predictive_experiment(X_gt,
                                                        X_syns,
                                                        workspace_folder=workspace_folder,
                                                        results_folder=results_folder,
                                                        save=save,
                                                        load=load,
                                                        plot=True,
                                                        )

                
                means, std = model_evaluation_experiment(X_gt, X_syns, workspace_folder=workspace_folder, relative='',
                                                    model_type='deepish_mlp',
                                                    load=load,
                                                    save=load,
                                                    verbose=verbose

                )

                means, std = model_evaluation_experiment(X_gt, X_syns, workspace_folder=workspace_folder, relative='',
                                                    model_type='mlp',
                                                    load=load,
                                                    save=load,
                                                    verbose=verbose

                )

                