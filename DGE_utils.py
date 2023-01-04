import xgboost
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error, log_loss, brier_score_loss
from sklearn.model_selection import KFold, train_test_split
import sklearn

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils import reproducibility

from bnaf.toy2d import main as bnaf
from bnaf.toy2d import compute_log_p_x

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import torch


def cat_dl(X_syns, n_limit=None):
    """
    Concatenate a list of GenericDataLoader objects into one GenericDataLoader object
    """
    if n_limit is not None:
        X_syn_cat = pd.concat([X_syns[i][:n_limit]
                              for i in range(len(X_syns))], axis=0)
    else:
        X_syn_cat = pd.concat([X_syns[i].dataframe()
                              for i in range(len(X_syns))], axis=0)
    X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
    X_syn_cat.targettype = X_syns[0].targettype
    return X_syn_cat


def parallel_for(func, args_list, max_workers=4):
    from concurrent.futures import ThreadPoolExecutor

    # Create a ThreadPoolExecutor with the desired number of threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use the executor to map the function over the list of arguments
        results = executor.map(func, args_list)

    return results


def init_model(model_type, targettype):
    """
    Initialize a model of the given type.
    """
    if model_type == 'lr':
        if targettype == 'classification':
            model = sklearn.linear_model.LogisticRegression()
        else:
            model = sklearn.linear_model.LinearRegression()
    elif model_type == 'mlp':
        if targettype == 'classification':
            model = sklearn.neural_network.MLPClassifier()
        else:
            model = sklearn.neural_network.MLPRegressor()
    elif model_type == 'deep_mlp':
        if targettype == 'classification':
            model = sklearn.neural_network.MLPClassifier(
                hidden_layer_sizes=(100, 100, 100))
        else:
            model = sklearn.neural_network.MLPRegressor(
                hidden_layer_sizes=(100, 100, 100))

    elif model_type == 'rf':
        if targettype == 'classification':
            model = sklearn.ensemble.RandomForestClassifier()
        else:
            model = sklearn.ensemble.RandomForestRegressor()
    elif model_type == 'knn':
        if targettype == 'classification':
            model = sklearn.neighbors.KNeighborsClassifier()
        else:
            model = sklearn.neighbors.KNeighborsRegressor()
    elif model_type == 'svm':
        if targettype == 'classification':
            model = sklearn.svm.SVC()
        else:
            model = sklearn.svm.SVR()
    elif model_type == 'xgboost':
        if targettype == 'classification':
            model = xgboost.XGBClassifier()
        else:
            model = xgboost.XGBRegressor()
    else:
        raise ValueError('Unknown model type')
    return model


def supervised_task(X_gt, X_syn, model=None, model_type='mlp', verbose=False):

    if type(model) == str or model is None:
        model = init_model(model_type, X_syn.targettype)
        X, y = X_syn.unpack(as_numpy=True)
        model.fit(X, y.reshape(-1, 1))
    if X_gt.targettype == 'regression':
        pred = model.predict(X_gt.unpack(as_numpy=True)[0])
    else:
        pred = model.predict_proba(X_gt.unpack(as_numpy=True)[0])[:, 1]    
    return pred, model


def compute_metrics(y_test, yhat_test, targettype='classification'):
    if targettype == 'classification':
        y_test = y_test.astype(bool)
        yhat_test = yhat_test.astype(float) 
        metrics = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'nll', 'brier',]
        scores = [roc_auc_score(y_test, yhat_test), accuracy_score(y_test, yhat_test>0.5),
                  f1_score(y_test, yhat_test>0.5), precision_score(
                      y_test, yhat_test>0.5), recall_score(y_test, yhat_test>0.5),
                  log_loss(y_test, yhat_test), brier_score_loss(y_test, yhat_test)]
    elif targettype == 'regression':
        metrics = ['mse', 'mae']
        scores = [mean_squared_error(
            y_test, yhat_test), mean_absolute_error(y_test, yhat_test)]
    else:
        raise ValueError('unknown target type')

    scores = np.array(scores).reshape(1, -1)
    scores = pd.DataFrame(scores, columns=metrics)
    return scores


def tt_predict_performance(X_test, X_train, model=None, model_type='mlp', verbose=False):
    """compute train_test performance for different metrics"""
    # import metrics

    x_train, y_train = X_train.unpack(as_numpy=True)
    x_test, y_test = X_test.unpack(as_numpy=True)

    if model is None:
        model = init_model(model_type, X_test.targettype)
        model.fit(x_train, y_train)

    yhat_test = model.predict(x_test)
    scores = compute_metrics(y_test, yhat_test, X_test.targettype)
    return scores, model


def aggregate_predictive(X_gt, X_syns, task=tt_predict_performance, models=None, task_type='', workspace_folder='workspace', results_folder='results', load=True, save=True,
                         approach='us', relative=False, run_for_all=True, verbose=False):
    """
    aggregate predictions from different synthetic datasets
    """

    results = []
    stds = []
    trained_models = []
    filename = ''
    fileroot = f'{workspace_folder}/{task.__name__}_{task_type}'
    if not os.path.exists(fileroot) and save:
        os.makedirs(fileroot)

    if run_for_all:
        range_limit = len(X_syns)
    else:
        range_limit = 1
    for i in range(range_limit):
        if models is None:
            if verbose:
                print(f'Saving model as {fileroot}_{filename}{i}.pkl')

            if os.path.exists(f'{fileroot}_{filename}{i}.pkl') and load:
                model = pickle.load(open(f"{fileroot}_{filename}{i}.pkl", "rb"))
            else:
                model = None
                print(f'Train model {i+1}/{len(X_syns)}')
        else:
            model = models[i]
        reproducibility.enable_reproducible_results(seed=i+2022)
        X_train = X_syns[i].train()
        if approach == 'naive':
            X_test = X_syns[i].test()
        elif approach == 'dge':
            X_syns_not_i = [X_syns[j] for j in range(len(X_syns)) if j != i]
            X_syns_not_i[0].targettype = X_syns[0].targettype
            X_test = cat_dl(X_syns_not_i)
        elif approach == 'dge_new':
            X_syns_not_i = [X_syns[j] for j in range(len(X_syns)) if j != i]
        elif approach == 'oracle':
            X_test = X_gt.test()
        else:
            raise ValueError('Unknown approach')

        if not approach == 'us_new':
            X_test.targettype = X_syns[0].targettype
            X_train.targettype = X_syns[0].targettype

            res, model = task(X_test, X_train, model, task_type, verbose)

            if relative and approach != 'oracle':
                X_test = X_gt.test()
                X_test.targettype = X_syns[0].targettype
                res_oracle, model = task(X_test, X_train, model, task_type, verbose)

                if relative == 'l2':
                    res = (res - res_oracle)**2
                elif relative == 'l1':
                    res = (res - res_oracle).abs()
                else:
                    raise ValueError('Unknown relative metric')

        else:
            if relative:
                raise ValueError('Relative not implemented for us_new')
            res = []
            for j in range(len(X_syns_not_i)):
                X_test = X_syns_not_i[j].test()
                X_test.targettype = X_syns[0].targettype
                X_train.targettype = X_syns[0].targettype
                res.append(task(X_test, X_train, model, task_type, verbose)[0])
            res, std = meanstd(pd.concat(res, axis=0))
            res = res
            stds.append(std)

        results.append(res)
        trained_models.append(model)
        # save model to disk as pickle
        if models is None and save:
            pickle.dump(model, open(f"{fileroot}_{filename}{i}.pkl", "wb"))

    results = pd.concat(results, axis=0)
    if approach != 'us_new':
        return *meanstd(results), trained_models
    else:
        stds = pd.concat(stds, axis=0)
        stds = stds.mean(axis=0).to_frame().T
        means, stds2 = meanstd(results)
        return means, (stds**2+stds2**2)**0.5, trained_models


# def cv_predict_performance(X_gt, X_syn, model=None, model_type='mlp', n_splits=5, verbose=False):
#     """compute cross-validated performance for different metrics"""

#     # initialize KFold with fixed random state for reproducibility
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     x, y = X_syn.unpack(as_numpy=True)

#     if X_syns[0].targettype == 'classification':
#         metrics = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']
#     elif X_syns[0].targettype == 'regression':
#         metrics = ['r2', 'mse', 'mae']
#     else:
#         raise ValueError('unknown target type')

#     models = model

#     scores = []
#     model_list = []

#     for i, train_index, test_index in zip(range(n_splits), kf.split(x, y)):
#         x_train, x_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         if models is None:
#             model = init_model(model_type, X_syns[0].targettype)
#             model.fit(x_train, y_train)
#         else:
#             model = models[i]

#         yhat_test = model.predict(x_test)

#         if X_syns[0].targettype == 'classification':
#             scores.append([roc_auc_score(y_test, yhat_test), accuracy_score(y_test, yhat_test), f1_score(
#                 y_test, yhat_test), precision_score(y_test, yhat_test), recall_score(y_test, yhat_test)])
#         elif X_syns[0].targettype == 'regression':
#             scores.append([r2_score(y_test, yhat_test), mean_squared_error(
#                 y_test, yhat_test), mean_absolute_error(y_test, yhat_test)])

#         if models is None:
#             model_list.append(model)

#     if models is None:
#         models = model_list

#     scores = np.array(scores)
#     # scores = np.concatenate((np.mean(scores, axis=0), np.std(scores, axis=0)),axis=1)
#     # metrics = metrics + [f'{m}_std' for m in metrics]
#     scores = pd.DataFrame(scores, columns=metrics)
#     return scores, models


def meanstd(A):
    if type(A) == pd.DataFrame:
        return A.mean(axis=0).to_frame().T, A.std(axis=0).to_frame().T
    else:
        return np.mean(A, axis=0), np.std(A, axis=0)


def density_estimation(X_gt, X_syn, model=None, model_type='kde', verbose=False):
    """
    compute density estimation of X_syn, evaluate on X_gt
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from scipy.stats import gaussian_kde
    if model_type == 'bnaf':

        if model is None:
            if verbose:
                print('Training BNAF')
            model = bnaf(X_syn.unpack(as_numpy=True)[0])
        X = torch.utils.data.TensorDataset(torch.tensor(
            X_gt.unpack(as_numpy=True)[0], device=device).to(torch.float32))
        X = torch.utils.data.DataLoader(X, batch_size=10000, shuffle=False)
        prob = torch.cat(
            [
                torch.exp(compute_log_p_x(model, x_mb)).detach()
                for x_mb, in X
            ],
            0,
        )
        return torch.exp(prob).detach().cpu().numpy(), model
    elif model_type == 'kde':
        if model is None:
            if verbose:
                print('Training KDE')
            model = gaussian_kde(X_syn.unpack(as_numpy=True)[0].T)
        return model.pdf(X_gt.unpack(as_numpy=True)[0].T), model


def aggregate(X_gt, X_syns, task, models=None, task_type='', load=True, save=True, workspace_folder='workspace', filename='', verbose=False):
    """
    aggregate predictions from different synthetic datasets
    """

    results = []
    trained_models = []
    fileroot = f'{workspace_folder}/{task.__name__}_{task_type}'
    if not os.path.exists(fileroot) and save:
        os.makedirs(fileroot)

    for i in range(len(X_syns)):
        if models is None:
            if verbose:
                print(f'Saving model as {fileroot}_{filename}{i}.pkl')

            if os.path.exists(f'{fileroot}_{filename}{i}.pkl') and load:
                model = pickle.load(open(f"{fileroot}_{filename}{i}.pkl", "rb"))
            else:
                model = None
                if verbose:
                    print(f'Train model {i+1}/{len(X_syns)}')
        else:
            model = models[i]
        reproducibility.enable_reproducible_results(seed=i+2022)
        res, model = task(X_gt, X_syns[i], model, task_type, verbose)
        results.append(res)
        trained_models.append(model)
        # save model to disk as pickle
        if models is None and save:
            pickle.dump(model, open(f"{fileroot}_{filename}{i}.pkl", "wb"))

    return *meanstd(results), trained_models


def tsne(X):
    """
    Perform t-SNE dimensionality reduction to two dimensions
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    return X_2d


def aggregate_imshow(X_gt, X_syns, task, models=None, task_type='', results_folder='results', workspace_folder='workspace', load=True, save=True, filename=''):
    """
    Aggregate and plot predictions from different synthetic datasets, on a 2D space. E.g., density estimation, predictions.
    """
    xmin, ymin = np.min(X_gt.train().unpack(as_numpy=True)[0], axis=0)*1.05
    xmax, ymax = np.max(X_gt.train().unpack(as_numpy=True)[0], axis=0)*1.05

    steps = 400
    X_grid = np.linspace(xmin, xmax, steps)
    Y_grid = np.linspace(ymin, ymax, steps)

    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    X_grid = pd.DataFrame(np.c_[X_grid.ravel(), Y_grid.ravel()])
    X_grid['target'] = -1
    X_grid = GenericDataLoader(X_grid, target_column="target", train_size=0.01)
    X_grid.targettype = X_syns[0].targettype

    y_pred_mean, y_pred_std, models = aggregate(
        X_grid, X_syns,
        task=task,
        models=models,
        task_type=task_type,
        load=load,
        save=save,
        workspace_folder=workspace_folder,
        filename=f'n{len(X_gt.train().unpack()[0])}_{filename}')

    for y, stat in zip((y_pred_mean, y_pred_std), ('mean', 'std')):
        plt.figure(figsize=(8, 6), dpi=100)
        plt.imshow(y.reshape(steps, steps)[::-1],
                   cmap='viridis', extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        if X_gt.dataset == 'gaussian':
            plt.vlines(0, ymin, ymax, colors='r', linestyles='dashed')
        filename = f'{results_folder}/{task.__name__}_n{len(X_gt.train().unpack()[0])}_{filename}{stat}'
        print(f'Saving {filename}.png')
        plt.savefig(filename+'.png')
        plt.show()
        X_train, y_train = X_gt.train().unpack(as_numpy=True)
        if len(np.unique(y_train)) == 2:
            plt.imshow(y.reshape(steps, steps)[
                       ::-1], cmap='viridis', extent=[xmin, xmax, ymin, ymax])
            y_train = y_train.astype(bool)
            plt.scatter(X_train[y_train, 0], X_train[y_train, 1], c='k', marker='.')
            plt.scatter(X_train[~y_train, 0], X_train[~y_train, 1], c='w', marker='.')
            plt.colorbar()
            if X_gt.dataset == 'gaussian':
                plt.vlines(0, ymin, ymax, colors='r', linestyles='dashed')
            plt.savefig(f'{filename}_with_samples.png')

            plt.show()

    return y_pred_mean, y_pred_std, models
