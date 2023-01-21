import xgboost
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, log_loss, brier_score_loss
from sklearn.model_selection import KFold, train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor    
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

from mpl_toolkits.axes_grid1 import make_axes_locatable

from hashlib import sha256

def hash_str2int(s):
    """
    Hash a string to an integer
    """
    
    return int(sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)


def accuracy_confidence_curve(y_true, y_prob, n_bins=20):
    thresholds = np.linspace(0.5, 0.95, n_bins)
    y_pred = y_prob > 0.5
    correct = y_true == y_pred
    accs = np.zeros(n_bins)
    
    for i, threshold in enumerate(thresholds):
        y_select = (y_prob > threshold) + (y_prob < (1 - threshold))
        accs[i] = np.mean(correct[y_select])
        
    return thresholds, accs

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
            model = MLPClassifier(hidden_layer_sizes=(100))
        else:
            model = MLPRegressor(hidden_layer_sizes=(100))
    elif model_type == 'deepish_mlp':
        if targettype == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 100))
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 100))
    
    elif model_type == 'deep_mlp':
        if targettype == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 100, 100))
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 100, 100))

    elif model_type == 'rf':
        # default 100 trees
        if targettype == 'classification':
            model = sklearn.ensemble.RandomForestClassifier()
        else:
            model = sklearn.ensemble.RandomForestRegressor()
    elif model_type == 'knn':
        # default 5 neighbors
        if targettype == 'classification':
            model = sklearn.neighbors.KNeighborsClassifier()
        else:
            model = sklearn.neighbors.KNeighborsRegressor()
    elif model_type == 'svm':
        # default rbf kernel
        if targettype == 'classification':
            model = sklearn.svm.SVC(probability=True)
        else:
            model = sklearn.svm.SVR()
    elif model_type == 'xgboost':
        if targettype == 'classification':
            model = xgboost.XGBClassifier()
        else:
            model = xgboost.XGBRegressor()
    else:
        raise ValueError('Unknown model type')
    
    # Wrap the model in a pipeline to scale the data
    #! Add scaling only of continuous and encoding of categorical
    model = Pipeline([('scaler', StandardScaler()), ('model', model)])
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


def roc_auc_score_rob(y_true, y_score, throw_error_if_nan=True):
    """
    Robust version of sklearn.metrics.roc_auc_score
    """
    if len(np.unique(y_true))>1 or throw_error_if_nan:
        roc_auc_score(y_true, y_score)
    else:
        return np.nan

def compute_metrics(y_test, yhat_test, targettype='classification'):
    if targettype == 'classification':
        y_test = y_test.astype(bool)
        yhat_test = yhat_test.astype(float) 
        metrics = ['AUC', 'Acc', 'F1', 'Precision', 'Recall', 'NLL', 'Brier',]
        scores = [roc_auc_score(y_test, yhat_test), accuracy_score(y_test, yhat_test>0.5),
                  f1_score(y_test, yhat_test>0.5), precision_score(
                      y_test, yhat_test>0.5), recall_score(y_test, yhat_test>0.5),
                  log_loss(y_test, yhat_test, labels=[0,1]), brier_score_loss(y_test, yhat_test)]
    elif targettype == 'regression':
        metrics = ['RMSE', 'MAE']
        scores = [np.sqrt(mean_squared_error(
            y_test, yhat_test)), mean_absolute_error(y_test, yhat_test)]
    else:
        raise ValueError('unknown target type')
    
    #scores = np.round(scores, 3)
    scores = np.array(scores).reshape(1, -1)
    scores = pd.DataFrame(scores, columns=metrics)
    return scores


def outlier_compute(X):
    center = X.unpack(as_numpy=True)[0].mean(axis=0)
    dis_to_center = lambda x: np.sum((x.unpack(as_numpy=True)[0]-center)**2,axis=1)
    threshold = np.quantile(dis_to_center(X), 0.9)
    
    def subset(X):
        Xout = X[dis_to_center(X)>threshold]
        if type(Xout) == pd.DataFrame:
            Xout = GenericDataLoader(Xout, target='target')
            if hasattr(X, 'targettype'):
                Xout.targettype = X.targettype
        return Xout
    
    return subset


def tt_predict_performance(X_test, X_train, model=None, model_type='mlp', subset=None, verbose=False):
    """compute train_test performance for different metrics"""
    # import metrics

    x_train, y_train = X_train.unpack(as_numpy=True)
    if subset is not None:
        X_test = subset(X_test)
    x_test, y_test = X_test.unpack(as_numpy=True)
    
    if model is None:
        model = init_model(model_type, X_test.targettype)
        model.fit(x_train, y_train)

    if X_test.targettype == 'regression':
        yhat_test = model.predict(x_test)
    else:
        yhat_test = model.predict_proba(x_test)[:, 1]
    
    scores = compute_metrics(y_test, yhat_test, X_test.targettype)
    return scores, model


def aggregate_predictive(X_gt, X_syns, task=tt_predict_performance, models=None, task_type='', workspace_folder=None, results_folder=None, load=True, save=True,
                         approach='DGE', relative=False, run_for_all=True, verbose=False, K=None, subset=None):
    """
    aggregate predictions from different synthetic datasets
    """

    results = []
    stds = []
    trained_models = []
    fileroot = os.path.join(workspace_folder, f'model_eval_{task_type}')
    
    if K is None:
        K = len(X_syns)
    
    if not os.path.exists(fileroot) and save:
        os.makedirs(fileroot)
    
    if run_for_all:
        range_limit = len(X_syns)
    else:
        range_limit = 1
    
    for i in range(range_limit):
        filename = f'{fileroot}_{i}.pkl'
        if models is None:
            if os.path.exists(filename) and load:
                model = pickle.load(open(filename, "rb"))
            else:
                model = None
                if verbose:
                    print(f'Train model {i+1}/{len(X_syns)} and save as {filename}')
        else:
            model = models[i]
        reproducibility.enable_reproducible_results(seed=i+2022)
        X_train = X_syns[i].train()
        if approach == 'Naive':
            X_test = X_syns[i].test()
        elif 'alternative' in approach:
            X_syns_not_i = [X_syns[j] for j in range(len(X_syns)) if j != i][:K-1]
        elif 'DGE' in approach:
            X_syns_not_i = [X_syns[j] for j in range(len(X_syns)) if j != i][:K-1]
            X_syns_not_i[0].targettype = X_syns[0].targettype
            X_test = cat_dl(X_syns_not_i)
        elif approach == 'Oracle':
            X_test = X_gt.test()
        else:
            raise ValueError('Unknown approach')

        if not 'alternative' in approach:
            X_test.targettype = X_syns[0].targettype
            X_train.targettype = X_syns[0].targettype

            res, model = task(X_test, X_train, model, task_type, subset=subset, verbose=verbose)

            if relative and approach != 'Oracle':
                X_test = X_gt.test()
                X_test.targettype = X_syns[0].targettype
                res_oracle, model = task(X_test, X_train, model, task_type, subset=subset, verbose=verbose)

                if relative in ['l2']:
                    res = (res - res_oracle)**2
                elif relative == 'l1':
                    res = (res - res_oracle).abs()
                else:
                    raise ValueError('Unknown relative metric')

        else:
            if relative:
                raise ValueError('Relative not implemented for DGE_alternative')
            res = []
            for j in range(len(X_syns_not_i)):
                X_test = X_syns_not_i[j].test()
                X_test.targettype = X_syns[0].targettype
                X_train.targettype = X_syns[0].targettype
                res.append(task(X_test, X_train, model, task_type, subset=subset, verbose=verbose)[0])
            res, std = meanstd(pd.concat(res, axis=0))
            stds.append(std)

        results.append(res)
        trained_models.append(model)
        # save model to disk as pickle
        if models is None and save:
            pickle.dump(model, open(filename, "wb"))

    results = pd.concat(results, axis=0)
    if approach != 'DGE_alternative':
        return *meanstd(results), trained_models, results
    else:
        stds = pd.concat(stds, axis=0)
        stds = stds.mean(axis=0).to_frame().T
        means, stds2 = meanstd(results)
        return means, (stds**2+stds2**2)**0.5, trained_models, None


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


def aggregate(X_gt, X_syns, task, models=None, task_type='', load=True, save=True, workspace_folder=None, filename='', verbose=False):
    """
    aggregate predictions from different synthetic datasets
    """

    results = []
    trained_models = []
    fileroot = f'{workspace_folder}/{task.__name__}_{task_type}'
    
    if (save or load) and not os.path.exists(fileroot):
        os.makedirs(fileroot)

    for i in range(len(X_syns)):
        full_filename = f'{fileroot}_{filename}_{i}.pkl'
        if models is None:
            if verbose:
                print(f'Saving model as {full_filename}')

            if os.path.exists(full_filename) and load:
                model = pickle.load(open(full_filename, "rb"))
            else:
                model = None
                if verbose:
                    print(f'Train model {i+1}/{len(X_syns)}')
                seed = hash_str2int(full_filename)
                reproducibility.enable_reproducible_results(seed=seed)
        else:
            model = models[i]

        res, model = task(X_gt, X_syns[i], model, task_type, verbose)
        results.append(res)
        trained_models.append(model)
        # save model to disk as pickle
        if models is None and save:
            pickle.dump(model, open(full_filename, "wb"))

    return *meanstd(results), trained_models


def tsne(X):
    """
    Perform t-SNE dimensionality reduction to two dimensions
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    return X_2d


def aggregate_imshow(X_gt, X_syns, task, models=None, task_type='', results_folder=None, workspace_folder='workspace', load=True, save=True, filename='', baseline_contour=None):
    """
    Aggregate and plot predictions from different synthetic datasets, on a 2D space. E.g., density estimation, predictions.
    """

    xmin = ymin = np.min(X_gt.train().unpack(as_numpy=True)[0])
    xmax = ymax = np.max(X_gt.train().unpack(as_numpy=True)[0])
    
    steps = 400
    X_grid = np.linspace(xmin, xmax, steps)
    Y_grid = np.linspace(ymin, ymax, steps)

    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    Z_grid = pd.DataFrame(np.c_[X_grid.ravel(), Y_grid.ravel()])
    Z_grid['target'] = -1
    Z_grid = GenericDataLoader(Z_grid, target_column="target", train_size=0.01)
    Z_grid.targettype = X_syns[0].targettype

    y_pred_mean, y_pred_std, models = aggregate(
        Z_grid, X_syns,
        task=task,
        models=models,
        task_type=task_type,
        load=load,
        save=save,
        workspace_folder=workspace_folder,
        filename=filename)

    contour = [X_grid, Y_grid, y_pred_mean.reshape(steps, steps)]
        
    for y, stat in zip((y_pred_mean, y_pred_std), ('mean', 'std')):
        fig = plt.figure(figsize=(3, 2.5), dpi=300, tight_layout=True)
        ax = plt.axes()
        if 'oracle' in filename.lower():
            plt.contour(X_grid, Y_grid, contour[2], levels = [0.5], colors='w', linestyles=':')
        else:
            plt.contour(X_grid, Y_grid, contour[2], levels = [0.5], colors='r', linestyles='--')

        if baseline_contour is not None:
            plt.contour(baseline_contour[0], baseline_contour[1], baseline_contour[2], levels = [0.5], colors='w', linestyles=':')

        im = ax.imshow(y.reshape(steps, steps),
                   cmap='viridis', extent=[xmin, xmax, ymin, ymax], origin='lower')

        ax.set_aspect('equal', 'box')
        
        # if 'gaussian' in results_folder:
        #     plt.vlines(0, ymin, ymax, colors='r', linestyles='dashed')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax) 
        
        
        if save:
            filename_base = results_folder+f'{task.__name__}_{task_type}_{filename}'
            filename_full = filename_base+stat+'.png'
            print(f'Saving {filename_full}')
            fig.savefig(f'{filename_full}', bbox_inches="tight")
        
        plt.show()

        X_train, y_train = X_gt.train().unpack(as_numpy=True)
            
    if len(np.unique(y_train)) == 2 and 'oracle' in filename.lower():
        fig = plt.figure(figsize=(3, 2.5), dpi=300, tight_layout=True)
        ax = plt.axes()
        ax.set_aspect('equal', 'box')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        y_train = y_train.astype(bool)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.')
        plt.tight_layout()
        # if 'gaussian' in results_folder:
        #     plt.vlines(0, ymin, ymax, colors='r', linestyles='dashed')
        

        if save:
            fig.savefig(f'{filename_base}_samples.png', bbox_inches='tight')
            
        plt.show()

    return y_pred_mean, y_pred_std, models, contour



######## paper/results/ formatting

def get_folder_names(dataset, model_name, max_n, nsyn):
    workspace_folder = os.path.join(
        "workspace", dataset, model_name, f'nmax_{max_n}_nsyn_{nsyn}')
    results_folder = os.path.join(
        "uncertainty_results", f'{dataset}_{model_name}_nmax_{max_n}_nsyn_{nsyn}')
    return workspace_folder, results_folder


def mean_across_pandas(dfs, precision=3):
    dfs = pd.concat(dfs.values())
    df_mean = dfs.groupby(level=0).mean()
    df_mean = df_mean.round(precision)
    print(df_mean.to_latex(float_format=lambda x: f'%.{precision}f' % x))
    return df_mean

# use scores but report per dataset
def metric_different_datasets(dfs, metric='AUC', to_print=True, precision=3):
    df_all_datasets = pd.concat([score[metric] for score in dfs.values()], axis=1)
    df_all_datasets.columns = dfs.keys()
    df_all_datasets.round(precision)

    for dataset in zip(['moons', 'circles', 'adult', 'breast_cancer', 'seer', 'covid'][::-1],
                       ['Moons', 'Circles', 'Adult Income', 'Breast Cancer', 'SEER', 'COVID-19'][::-1]):
        try:
            df_all_datasets.insert(0, dataset[1], df_all_datasets.pop(dataset[0]))
        except:
            continue
    df_all_datasets['Mean'] = df_all_datasets.mean(axis=1)

    if to_print:
        print(df_all_datasets.to_latex(float_format=lambda x: f'%.{precision}f' % x))

    return df_all_datasets


def add_std(df, std, precision=3):
    formatting_string = '{.'+str(precision)+'f'
    df = df.round(precision)
    std = std.round(precision)
    df.style.format(precision=3, formatter=formatting_string)
    std.style.format(precision=3, formatter=formatting_string)
    df= df.astype(str)
    std = std.astype(str)
    for column in df.columns:
        if column in std.columns:
            df[column] = df[column] + ' ± ' + std[column]
    return df

