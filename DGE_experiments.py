import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import os

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils import reproducibility

from DGE_utils import supervised_task, aggregate_imshow, aggregate, density_estimation, aggregate_predictive, cat_dl

############################################################################################################
# Model training. Predictive performance


def predictive_experiment(X_gt, X_syns, task_type='mlp', results_folder='results', workspace_folder='workspace', load=True, save=True):
    """Compares predictions by different approaches.

    Args:
        X_test (GenericDataLoader): Test data.
        X_syns (List(GenericDataLoader)): List of synthetic datasets.
        X_gt (GenericDataLoader): Real data
        load (bool, optional): Load results, if available. Defaults to True.
        save (bool, optional): Save results when done. Defaults to True.
    """

    X_test = X_gt.test()

    # DGE (k=20)
    if len(X_syns) != 20:
        raise ValueError('X_syns assumed to have 20 elements in this experiment.')

    if X_syns[0].targettype is not None:
        if X_gt.shape[1] == 2:
            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syns, supervised_task, models=None, workspace_folder=workspace_folder, results_folder=results_folder, task_type=task_type, load=load, save=save)
        else:
            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syns, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save)

        y_preds = [y_pred_mean]

    # DGE (k=10)
    if X_syns[0].targettype is not None:
        if X_gt.shape[1] == 2:
            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syns[:5], supervised_task, models=None, task_type=task_type, workspace_folder=workspace_folder, results_folder=results_folder, load=load, save=save)
        else:
            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syns, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save)

        y_preds.append(y_pred_mean)

    # DGE (k=5)
    if X_syns[0].targettype is not None:
        if X_gt.shape[1] == 2:
            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syns[:5], supervised_task, models=None, task_type=task_type, workspace_folder=workspace_folder, results_folder=results_folder, load=load, save=save)
        else:
            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syns, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save)

        y_preds.append(y_pred_mean)

    # Single model
    if X_syns[0].targettype is not None:
        X_syn_0 = [X_syns[0] for _ in range(len(X_syns))]
        if X_gt.shape[1] == 2:
            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syn_0, supervised_task, models=None, task_type=task_type, load=load, save=save, filename='naive')
        else:
            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syn_0, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='naive')

        y_preds.append(y_pred_mean)

    # Data aggregated

    if X_syns[0].targettype is not None:
        X_syn_cat = pd.concat([X_syns[i].dataframe()
                              for i in range(len(X_syns))], axis=0)
        X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
        X_syn_cat.targettype = X_syns[0].targettype
        X_syn_cat = [X_syn_cat for _ in range(len(X_syns))]
        #X_syn_cat = [X_syn_cat.sample(len(X_syns[0])) for _ in range(len(X_syns))]

        if X_gt.shape[1] == 2:
            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syn_cat, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='concat')
        else:
            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syn_cat, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='concat')

        y_preds.append(y_pred_mean)



    if X_syns[0].targettype is 'classification':
        # Consider calibration of different approaches
        plt.figure(figsize=(10, 10))
        for y_pred in y_preds:
            y_true = X_test.dataframe()['target'].values
            prob_true, prob_pred = calibration_curve(y_true, y_pred)
            plt.plot(prob_pred, prob_true)
        
        plt.xlabel = 'Mean predicted probability'
        plt.ylabel = 'Fraction of positives'

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.legend(['DGE (k=20)', 'DGE (k=10)', 'DGE (k=5)',
                    'Naive (single)', 'Naive (concat)', 'Perfect calibration'])

        
        if save:
            filename = os.path.join(results_folder, 'calibration_curve.png')
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            plt.savefig(filename)

        plt.show()


##############################################################################################################

# Model evaluation and selection experiments

def model_evaluation_experiment(X_gt, X_syns, model_type, relative=False, load=True, save=True):
    means = []
    stds = []
    approaches = ['oracle', 'naive', 'dge']
    for i, approach in enumerate(approaches):
        mean, std, _ = aggregate_predictive(
            X_gt, X_syns, models=None, task_type=model_type, load=load, save=save, approach=approach, relative=relative, verbose=False)
        means.append(mean)
        stds.append(std)

    means = pd.concat(means, axis=0)
    stds = pd.concat(stds, axis=0)
    means *= 100
    stds *= 100
    means = means.round(2)
    stds = stds.round(2)
    res = means.astype(str) + ' ± ' + stds.astype(str)
    res.index = approaches
    res.index.Name = 'Approach'
    return res, means, stds


def model_selection_experiment(X_gt, X_syns, relative='l1', metric='accuracy', load=True, save=True):
    model_types = ['lr', 'mlp', 'deep_mlp', 'rf', 'knn', 'svm', 'xgboost']
    metric = 'accuracy'
    results = []
    means = []
    relative = 'l1'
    for i, model_type in enumerate(model_types):
        res, mean, _ = model_evaluation_experiment(X_gt, X_syns, model_type, relative=relative, load=load, save=save)
        results.append(res[metric])
        means.append(mean[metric])

    means = pd.concat(means, axis=1)
    approaches = ['oracle', 'naive', 'DGE']
    means.index = approaches
    means.columns = model_types
    results = pd.concat(results, axis=1)
    results.columns = model_types

    # sort based on oracle
    sorting = [model_types[i] for i in means.loc['oracle'].argsort()]
    means = means.loc[:, sorting]
    results = results.loc[:, sorting]
    
    print(results)

    means_sorted = means.loc[:, sorting]

    for approach in approaches:
        sorting_k = means_sorted.loc[approach].argsort()
        sorting_k = sorting_k.argsort()
        means_sorted.loc[approach+' rank'] = sorting_k.astype(int)+1

    means_sorted.iloc[3:].astype(int)
    print(means_sorted)
    return results, means_sorted

##############################################################################################################

# Predictive uncertainty with varying number of synthetic data points


def predictive_varying_nsyn(X_gt, X_syns, dataset, model_name, n_models, nsyn, results_folder, workspace_folder, load=True, save=True, verbose=True):
    # Generative uncertainty
    # Let us first look at the generative estimates
    nsyn = X_syns[0].shape[0]
    n_syns = [nsyn//100, nsyn//10, nsyn]
    if X_syns[0].targettype is not None and X_gt.shape[1] == 2:
        for n_syn in n_syns:
            ### DGE (k=20)
            X_syns_red = [GenericDataLoader(
                X_syns[i][:n_syn], target_column='target') for i in range(len(X_syns))]
            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syns_red, supervised_task, models=None, task_type='mlp', load=load, save=save, filename=f'n_syn{n_syn}_dge')

            ### DGE (k=10)
            X_syns_red = [GenericDataLoader(
                X_syns[i][:n_syn], target_column='target') for i in range(10)]
            y_pred_mean, y_pred_std, _ = aggregate_imshow(
                X_gt, X_syns_red, supervised_task, models=models[:10], task_type='mlp', load=load, save=save, filename=f'n_syn{n_syn}_dge_k=10')

            ### DGE (k=5)
            X_syns_red = [GenericDataLoader(
                X_syns[i][:n_syn], target_column='target') for i in range(5)]
            y_pred_mean, y_pred_std, _ = aggregate_imshow(
                X_gt, X_syns_red, supervised_task, models=models[:5], task_type='mlp', load=load, save=save, filename=f'n_syn{n_syn}_dge_k=5')

            # Single model
            # Now let's look at the same behaviour by a single data and a downstream DE
            index = 0
            X_syn_0 = [GenericDataLoader(X_syns[index][:n_syn], target_column='target')
                       for i in range(len(X_syns))]
            y_pred_mean, y_pred_std, _ = aggregate_imshow(
                X_gt, X_syn_0, supervised_task, models=[models[index] for i in range(len(X_syn_0))], task_type='mlp', load=False, save=save, filename=f'n_syn{n_syn}_naive')

            # Aggregated data
            # And what happens when using all data for the downstream DE?
            X_syn_cat = cat_dl(X_syns, n_limit=n_syn)
            X_syn_cat = [X_syn_cat for _ in range(len(X_syns))]
            #X_syn_cat = [X_syn_cat.sample(len(X_syns[0])) for _ in range(len(X_syns))]

            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syn_cat, supervised_task, models=None, task_type='mlp', load=load, save=save, filename=f'n_syn{n_syn}_concat')


#############################################################################################################

# Density estimation of synthetic data outputs

def density_experiment(X_gt, X_syns, load=True, save=True):
    # Density estimation experiment
    # Approximate density of synthetic data outputs

    X_test = X_gt.test()
    X_test.targettype = X_syns[0].targettype
    y_pred_mean, y_pred_std, models = aggregate_imshow(
        X_test, X_syns, density_estimation, models=None, task_type='kde', load=load, save=save)
