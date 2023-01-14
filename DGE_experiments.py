from DGE_utils import outlier_compute
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import os

from synthcity.plugins.core.dataloader import GenericDataLoader

from DGE_utils import supervised_task, aggregate_imshow, aggregate, density_estimation, aggregate_predictive, cat_dl, compute_metrics, accuracy_confidence_curve

############################################################################################################
# Model training. Predictive performance


def predictive_experiment(X_gt, X_syns, task_type='mlp', results_folder=None, workspace_folder='workspace', load=True, save=True, plot=False, outlier=False, verbose=False):
    """Compares predictions by different approaches.

    Args:
        X_test (GenericDataLoader): Test data.
        X_syns (List(GenericDataLoader)): List of synthetic datasets.
        X_test (GenericDataLoader): Real data
        load (bool, optional): Load results, if available. Defaults to True.
        save (bool, optional): Save results when done. Defaults to True.

    Returns:

    """
    if save and results_folder is None:
        raise ValueError('results_folder must be specified when save=True.')

    X_test = X_gt.test()
    d = X_test.unpack(as_numpy=True)[0].shape[1]

    if outlier:
        subset = outlier_compute(X_gt)
        X_test = subset(X_test)
        plot = False

    X_test.targettype = X_gt.targettype

    if not X_gt.targettype in ['regression', 'classification']:
        raise ValueError('X_gt.targettype must be regression or classification.')

    y_preds_for_plotting = {}

    # DGE (k=5, 10, 20)
    n_models = 20  # maximum K
    num_runs = len(X_syns)//n_models

    if num_runs > 1 and verbose:
        print('Computing means and stds')

    Ks = [20, 10, 5]
    y_DGE_approaches = [f'DGE$_{K}$' for K in Ks]
    y_naive_approaches = ['Naive (single)', 'Naive (ensemble)']
    keys = y_DGE_approaches + y_naive_approaches + ['Oracle']
    y_preds = dict(zip(keys,[[] for _ in keys]))

    # Oracle
    X_oracle = X_gt.train()
    X_oracle.targettype = X_syns[0].targettype

    X_oracle = [X_oracle] * n_models
    # Oracle ensemble

    for run in range(num_runs):
        run_label = f'run_{run}'

        # DGE
        starting_dataset = run*n_models
        models = None
        for K, approach in zip(Ks, y_DGE_approaches):
            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syns[starting_dataset:starting_dataset+K], supervised_task, models=models, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'DGE_{run_label}_')

            if d == 2 and plot and run == 0:
                aggregate_imshow(
                    X_test, X_syns[starting_dataset:starting_dataset+K], supervised_task, models=models, workspace_folder=workspace_folder, results_folder=results_folder, task_type=task_type, load=load, save=save, filename=f'DGE_{run_label}_')

            y_preds[approach].append(y_pred_mean)

            # for plotting calibration and confidence curves later
            if run == 0 and plot:
                y_preds_for_plotting[f'DGE (K={K})'] = y_pred_mean

        # Single dataset single model
        for approach in y_naive_approaches:
            if approach == 'Naive (single)':
                X_syn_run = [X_syns[run]]
            else:
                X_syn_run = [X_syns[run]] * n_models

            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syn_run, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'naive_m{run}_')

            if run == 0 and plot and 'ensemble' in approach:
                if d == 2:
                    aggregate_imshow(X_test, X_syn_run, supervised_task, models=models, results_folder=results_folder,
                                     task_type=task_type, load=load, save=save, filename=f'naive_m{run}_')

                y_preds_for_plotting['Naive'] = y_pred_mean

            y_preds[approach].append(y_pred_mean)

        if False:
            # Data aggregated
            # X_syn_cat = pd.concat([X_syns[i].dataframe()
            #                         for i in range(len(X_syns))], axis=0)
            # X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
            # X_syn_cat.targettype = X_syns[0].targettype
            # X_syn_cat = [X_syn_cat]
            # #X_syn_cat = [X_syn_cat.sample(len(X_syns[0])) for _ in range(len(X_syns))]

            # y_pred_mean, y_pred_std, models = aggregate(
            #         X_test, X_syn_cat, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='concat')

            # if d == 2 and plot:
            #     aggregate_imshow(
            #         X_test, X_syn_cat, supervised_task, models=models, results_folder=results_folder, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='concat')

            # y_preds['Naive (concat)'] = y_pred_mean
            pass

        # Oracle single

        if False:
            y_pred_mean, _, models = aggregate(
                X_test, X_oracle, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='oracle')

            if d == 2 and plot:
                aggregate_imshow(
                    X_test, X_oracle, supervised_task, models=models, results_folder=results_folder, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='oracle')

            y_preds_for_plotting['Oracle (single)'] = y_pred_mean

        # Oracle ensemble

        y_pred_mean, _, models = aggregate(
            X_test, X_oracle, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'oracle_{run_label}_')

        if d == 2 and plot and run == 0:
            aggregate_imshow(
                X_test, X_oracle, supervised_task, models=models, results_folder=results_folder, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='oracle')

        if run == 0 and plot:
            y_preds_for_plotting['Oracle'] = y_pred_mean

        y_preds['Oracle'].append(y_pred_mean)


    # Evaluation
    ## Plotting

    y_true = X_test.dataframe()['target'].values
        
    if X_syns[0].targettype is 'classification' and plot:
        # Consider calibration of different approaches
        fig = plt.figure(figsize=(4, 4), tight_layout=False, dpi=200)
        for key, y_pred in y_preds_for_plotting.items():
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            plt.plot(prob_pred, prob_true, label=key)

        plt.xlabel = 'Mean predicted probability'
        plt.ylabel = 'Fraction of positives'

        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        plt.legend()

        if save:
            filename = results_folder+'_calibration_curve.png'
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            fig.savefig(filename, dpi=200)

        plt.close()

        plt.figure(figsize=(4, 4), dpi=200, tight_layout=False)
        for key, y_pred in y_preds_for_plotting.items():
            thresholds, prob_true = accuracy_confidence_curve(y_true, y_pred, n_bins=20)
            plt.plot(thresholds, prob_true, label=key)

        plt.xlabel = r'Confidence threshold \tau'
        plt.ylabel = r'Accuracy on examples \hat{y}'

        plt.legend()

        if save:
            filename = results_folder+'_confidence_accuracy_curve.png'
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            plt.savefig(filename, dpi=200)

        if plot:
            plt.show()

        plt.close()

    ## Compute metrics

    scores_mean = {}
    scores_std = {}

    for approach in y_preds.keys():
        scores = []
        for y_pred in y_preds[approach]:
            scores.append(compute_metrics(y_true, y_pred, X_test.targettype))

        scores = pd.concat(scores, axis=0)

        scores_mean[approach] = np.mean(scores, axis=0)
        scores_std[approach] = np.std(scores, axis=0)

    scores_mean = pd.DataFrame.from_dict(
        scores_mean, orient='index', columns=scores.columns)
    scores_std = pd.DataFrame.from_dict(
        scores_std, orient='index', columns=scores.columns)

    return scores_mean, scores_std

##############################################################################################################

# Model evaluation and selection experiments


def model_evaluation_experiment(X_gt, X_syns, model_type, relative=False, workspace_folder=None, load=True, save=True, outlier=False, verbose=False):
    means = []
    stds = []
    approaches = ['Oracle', 'Naive', 'DGE (K=5)', 'DGE (K=10)', 'DGE (K=20)']
    K = [None, None, 5, 10, 20]
    if outlier:
        subset = outlier_compute(X_gt)
    else:
        subset = None

    for i, approach in enumerate(approaches):
        if verbose:
            print('Approach: ', approach)
        folder = os.path.join(workspace_folder, approach)
        mean, std, _ = aggregate_predictive(
            X_gt, X_syns, models=None, task_type=model_type, workspace_folder=folder, load=load, save=save, approach=approach, relative=relative, subset=subset, verbose=verbose, K=K[i])
        means.append(mean)
        stds.append(std)

    means = pd.concat(means, axis=0)
    stds = pd.concat(stds, axis=0)
    if relative == 'rmse':
        means = np.sqrt(means)
    if relative != 'l2':
        means = means.round(3)
    stds = stds.round(3)

    means.index = approaches
    stds.index = approaches
    means.index.Name = 'Approach'
    stds.index.Name = 'Approach'
    return means, stds


def model_selection_experiment(X_gt, X_syns, relative='l1', workspace_folder='workspace', load=True, save=True, outlier=False):
    model_types = ['lr', 'mlp', 'deep_mlp', 'rf', 'knn', 'svm', 'xgboost']
    all_stds = []
    all_means = []
    output_means = {}
    output_stds = {}

    for i, model_type in enumerate(model_types):
        mean, std = model_evaluation_experiment(
            X_gt, X_syns, model_type, workspace_folder=workspace_folder, relative=relative, load=load, save=save, outlier=outlier)
        all_means.append(mean)
        all_stds.append(std)

    for metric in mean.columns:
        means = []
        stds = []
        for i, model_type in enumerate(model_types):
            means.append(all_means[i][metric])
            stds.append(all_stds[i][metric])

        means = pd.concat(means, axis=1)
        stds = pd.concat(stds, axis=1)
        means.columns = model_types
        stds.columns = model_types
        approaches = mean.index
        means.index = approaches
        stds.index = approaches

        # sort based on oracle
        sorting = [model_types[i] for i in means.loc['Oracle'].argsort()]
        means_sorted = means.loc[:, sorting]
        stds_sorted = stds.loc[:, sorting]

        for approach in approaches:
            sorting_k = means_sorted.loc[approach].argsort()
            sorting_k = sorting_k.argsort()
            means_sorted.loc[approach+' rank'] = 7-sorting_k.astype(int)

        means_sorted.iloc[3:].astype(int)

        output_means[metric] = means_sorted
        output_stds[metric] = stds_sorted

    return output_means, output_stds

# def model_predictive_uncertainty_experiment(X_gt, X_syns, model_type, workspace_folder=None, results_folder=None, load=True, save=True):

#     if save and (results_folder is None or workspace_folder is None):
#         raise ValueError('Please provide a workspace and results folder')
#     if load and workspace_folder is None:
#         raise ValueError('Please provide a workspace folder')


##############################################################################################################

# Predictive uncertainty with varying number of synthetic data points


def predictive_varying_nsyn(X_gt, X_syns, dataset, model_name, nsyn, results_folder, workspace_folder, load=True, save=True, verbose=True):
    # Generative uncertainty
    # Let us first look at the generative estimates
    nsyn = X_syns[0].shape[0]
    n_syns = [500, 1000, 2000, 5000, 10000, 20000]
    if X_syns[0].targettype is not None and X_gt.shape[1] == 2:
        for n_syn in n_syns:
            # DGE (k=20)
            X_syns_red = [GenericDataLoader(
                X_syns[i][:n_syn], target_column='target') for i in range(len(X_syns))]
            y_pred_mean, y_pred_std, models = aggregate_imshow(
                X_gt, X_syns_red, supervised_task, models=None, task_type='mlp', load=load, save=save, filename=f'n_syn{n_syn}_dge')

            # DGE (k=10)
            X_syns_red = [GenericDataLoader(
                X_syns[i][:n_syn], target_column='target') for i in range(10)]
            y_pred_mean, y_pred_std, _ = aggregate_imshow(
                X_gt, X_syns_red, supervised_task, models=models[:10], task_type='mlp', load=load, save=save, filename=f'n_syn{n_syn}_dge_k=10')

            # DGE (k=5)
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
            # X_syn_cat = [X_syn_cat.sample(len(X_syns[0])) for _ in range(len(X_syns))]

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
