from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, load_digits, make_moons, make_circles, fetch_california_housing, fetch_covtype

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils import reproducibility
from synthcity.plugins import Plugins
import synthcity.logger as log

from data.dataloader_seer_cutract import load_seer_cutract
from data.dataloader_adult import load_adult_census
from data.dataloader_covid import load_covid


def load_real_data(dataset, p_train=0.8, max_n=None, reduce_to=20000):


    if dataset == 'diabetes':
        X, y = load_diabetes(return_X_y=True, as_frame=True)
    elif dataset == 'iris':
        X, y = load_iris(return_X_y=True, as_frame=True)
    elif dataset == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    elif dataset == 'wine':
        X, y = load_wine(return_X_y=True, as_frame=True)
    elif dataset == 'adult':
        X, y = load_adult_census(as_frame=True)
    elif dataset == 'covid':
        X, y = load_covid(reduce_to = reduce_to)
        
    elif dataset == 'digits':
        X, y = load_digits(return_X_y=True, as_frame=True)
    elif dataset == 'moons':
        X, y = make_moons(n_samples=10000, noise=0.2, random_state=0)
        X = pd.DataFrame(X)
    elif dataset == 'circles':
        X, y = make_circles(n_samples=10000, noise=0.3, factor=0.5, random_state=0)
        X = pd.DataFrame(X)
    elif dataset == 'gaussian':
        n_real = 40000
        X = np.random.randn(n_real, 2)
        X = pd.DataFrame(X)
        noise = 2
        y = X[0] > noise*(np.random.uniform(size=n_real)-1/2)
    elif dataset == 'cal_housing':
        X = fetch_california_housing()
        X, y = X.data, X.target
        X = pd.DataFrame(X)
    elif dataset=='covtype':
        
        X = fetch_covtype()
        X, y = X.data, X.target
        X = pd.DataFrame(X)
    elif dataset in ['seer', 'cutract']:
        
        X, y = load_seer_cutract(name=dataset, seed=0, reduce_to=reduce_to)
        X = pd.DataFrame(X)
    elif dataset in ['uniform', 'test']:
        n_real = 10000
        X = np.random.uniform(size=(n_real, 2))
        X = pd.DataFrame(X)
        y = X[0] > np.random.uniform(size=n_real)
    else:
        raise ValueError('Unknown dataset')

    X["target"] = y
    if max_n is not None and X.shape[0]*p_train > max_n:
        p_train = max_n/X.shape[0]
    X_gt = GenericDataLoader(X, target_column="target", train_size=p_train)

    if len(np.unique(y)) == 1:
        X_gt.targettype = None
    elif len(np.unique(y)) <= 10:
        X_gt.targettype = 'classification'
    else:
        X_gt.targettype = 'regression'

    return X_gt


def get_synthetic_data(X_gt,
                       model_name,
                       n_models,
                       nsyn,
                       data_folder,
                       load_syn=True,
                       save=True,
                       verbose=False):

    X_train = X_gt.train()
    n_train = X_train.shape[0]
    X_syns = []

    # generate synthetic data using ensemble. Change seeds across models
    for i in range(n_models):
        os.makedirs(data_folder, exist_ok=True)
        filename = f"{data_folder}/Xsyn_n{n_train}_seed{i}.pkl"

        # Load data from disk if it exists and load_syn is True
        if os.path.exists(filename) and load_syn:
            X_syn = pickle.load(open(filename, "rb"))

            if len(X_syn)<nsyn:
                # generate more data if nsyn is too small
                if verbose:
                    print('Generating more data, existing dataset is smaller than nsyn')
                X_syn = generate_synthetic(model_name, n_models, save, verbose, X_train, i, filename)
            
        else:
            # Otherwise generate new data
            if verbose:
                print('Generating new data, filename is', filename)
            X_syn = generate_synthetic(model_name, n_models, save, verbose, X_train, i, filename)
            
        X_syn = GenericDataLoader(X_syn[:nsyn], target_column="target")
        X_syn.targettype = X_gt.targettype
        X_syns.append(X_syn)

    if verbose:
        # plot what we generated and compare to real
        X_syn_all = np.concatenate([X_syns[i].unpack(as_numpy=True)[0]
                                    for i in range(len(X_syns))])
        a = X_syn_all[np.random.choice(X_syn_all.shape[0], 1000, replace=False)]
        plt.scatter(a[:, 0], a[:, 1], marker='.')
        b = X_gt.train().unpack(as_numpy=True)[0]
        plt.scatter(b[:, 0], b[:, 1], marker='.')

    return X_syns

def generate_synthetic(model_name, n_models, save, verbose, X_train, i, filename):
    if verbose:
        print(f"Training model {i+1}/{n_models}")

    reproducibility.enable_reproducible_results(seed=i)
    if '_deep' in model_name:
        syn_model = Plugins().get(model_name.replace('_deep', ''), discriminator_n_layers_hidden=3, generator_n_layers_hidden=3)
    elif '_shallow' in model_name:
        syn_model = Plugins().get(model_name.replace('_shallow', ''), discriminator_n_layers_hidden=1, generator_n_layers_hidden=1)
    elif '_smallest' in model_name:
        syn_model = Plugins().get(model_name.replace('_smallest', ''), discriminator_n_layers_hidden=1, generator_n_layers_hidden=1, generator_n_units_hidden=100, discriminator_n_units_hidden=100)
    else:
        syn_model = Plugins().get(model_name)
    syn_model.fit(X_train)
    X_syn = syn_model.generate(count=20000) # we won't need more in any experiment
            
            # save X_syn to disk as pickle
    if save:
        pickle.dump(X_syn, open(filename, "wb"))

    
    return X_syn


def get_real_and_synthetic(dataset,
                           nsyn=None,
                           p_train=0.8,
                           n_models=20,
                           model_name='ctgan',
                           load_syn=True,
                           save=True,
                           verbose=False,
                           max_n = 2000,
                           reduce_to=20000):

    X_gt = load_real_data(dataset, p_train=p_train, max_n=max_n, reduce_to=reduce_to)
    X_train, X_test = X_gt.train(), X_gt.test()

    X_train.targettype = X_gt.targettype
    X_test.targettype = X_gt.targettype
    X_gt.dataset = dataset
    n_train = X_train.shape[0]

    if nsyn == 'train' or nsyn is None:
        nsyn = n_train

    data_folder = os.path.join("synthetic_data",dataset,model_name)
    print('n_total', X_gt.shape[0], 'n_train:', n_train)

    # generate synthetic data for all number of training samples
    X_syns = get_synthetic_data(X_gt, model_name,
                                n_models=n_models,
                                nsyn=nsyn,
                                data_folder=data_folder,
                                load_syn=load_syn,
                                save=save,
                                verbose=verbose)

    for i in range(len(X_syns)):
        X_syns[i].dataset = dataset
        X_syns[i].targettype = X_gt.targettype
        X_syns[i].dataset = dataset

    if dataset == 'covid':
        X_gt['target'] = (X_gt['target']-1).astype(bool)
        for i in range(len(X_syns)):
            X_syns[i]['target'] = (X_syns[i]['target']-1).astype(bool)
    
    return X_gt, X_syns
