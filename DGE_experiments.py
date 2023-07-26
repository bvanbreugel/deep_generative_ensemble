# stdlib
import os
import pickle
from typing import Callable, List

from DGE_utils import (
    accuracy_confidence_curve,
    aggregate,
    aggregate_imshow,
    aggregate_predictive,
    cat_dl,
    compute_metrics,
    supervised_task,
    tt_predict_performance,
)

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold

# synthcity absolute
from synthcity.plugins.core.dataloader import GenericDataLoader

############################################################################################################
# Model training. Predictive performance


def predictive_experiment(
    X_gt,
    X_syns,
    task_type="mlp",
    results_folder=None,
    workspace_folder="workspace",
    load=True,
    save=True,
    plot=False,
    outlier=False,
    verbose=False,
    include_concat=False,
):
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
        raise ValueError("results_folder must be specified when save=True.")

    X_test = X_gt.test()
    d = X_test.unpack(as_numpy=True)[0].shape[1]

    if isinstance(outlier, Callable):
        print("Using subset for evaluation")
        subset = outlier
        X_test = subset(X_test)
        plot = False
    elif outlier:
        raise ValueError("outlier boolean is no longer supported")

    X_test.targettype = X_gt.targettype

    if X_gt.targettype not in ["regression", "classification"]:
        raise ValueError("X_gt.targettype must be regression or classification.")

    # DGE (k=5, 10, 20)
    n_models = 20  # maximum K
    num_runs = len(X_syns) // n_models

    if num_runs > 1 and verbose:
        print("Computing means and stds")

    Ks = [20, 10, 5]
    y_DGE_approaches = ["DGE$_{" + str(K) + "}$" for K in Ks]
    y_naive_approaches = ["Naive (S)", "Naive (E)"]
    keys = (
        ["Oracle"]
        + y_naive_approaches
        + y_DGE_approaches[::-1]
        + ["DGE$_{20}$ (concat)"]
    )
    y_preds = dict(zip(keys, [[] for _ in keys]))
    keys_for_plotting = ["Oracle", "Naive"] + y_DGE_approaches[::-1]
    if include_concat:
        keys_for_plotting += ["DGE$_{20}$ (concat)"]
    y_preds_for_plotting = dict(zip(keys_for_plotting, [None] * len(keys_for_plotting)))

    # Oracle
    X_oracle = X_gt.train()
    X_oracle.targettype = X_syns[0].targettype

    X_oracle = [X_oracle] * n_models

    # Oracle ensemble

    for run in range(num_runs):
        run_label = f"run_{run}"

        # Oracle ensemble

        y_pred_mean, _, models = aggregate(
            X_test,
            X_oracle,
            supervised_task,
            models=None,
            workspace_folder=workspace_folder,
            task_type=task_type,
            load=load,
            save=save,
            filename=f"oracle_{run_label}_",
        )

        if d == 2 and plot and run == 0:
            _, _, _, contour = aggregate_imshow(
                X_test,
                X_oracle,
                supervised_task,
                models=models,
                results_folder=results_folder,
                workspace_folder=workspace_folder,
                task_type=task_type,
                load=load,
                save=save,
                filename="oracle",
            )

        if run == 0 and plot:
            y_preds_for_plotting["Oracle"] = y_pred_mean

        y_preds["Oracle"].append(y_pred_mean)

        # Single dataset single model
        for approach in y_naive_approaches:
            if approach == "Naive (S)":
                X_syn_run = [X_syns[run]]
            else:
                X_syn_run = [X_syns[run]] * n_models

            y_pred_mean, y_pred_std, models = aggregate(
                X_test,
                X_syn_run,
                supervised_task,
                models=None,
                workspace_folder=workspace_folder,
                task_type=task_type,
                load=load,
                save=save,
                filename=f"naive_m{run}_",
            )

            if run == 0 and plot and approach == "Naive (E)":
                if d == 2:
                    aggregate_imshow(
                        X_test,
                        X_syn_run,
                        supervised_task,
                        models=models,
                        results_folder=results_folder,
                        task_type=task_type,
                        load=load,
                        save=save,
                        filename=f"naive_m{run}_",
                        baseline_contour=contour,
                    )

                y_preds_for_plotting["Naive"] = y_pred_mean

            y_preds[approach].append(y_pred_mean)

        # DGE
        starting_dataset = run * n_models
        models = None
        for K, approach in zip(Ks, y_DGE_approaches):
            y_pred_mean, y_pred_std, models = aggregate(
                X_test,
                X_syns[starting_dataset : starting_dataset + K],
                supervised_task,
                models=models,
                workspace_folder=workspace_folder,
                task_type=task_type,
                load=load,
                save=save,
                filename=f"DGE_{run_label}_",
            )

            if d == 2 and plot and run == 0:
                aggregate_imshow(
                    X_test,
                    X_syns[starting_dataset : starting_dataset + K],
                    supervised_task,
                    models=models,
                    workspace_folder=workspace_folder,
                    results_folder=results_folder,
                    task_type=task_type,
                    load=load,
                    save=save,
                    filename=f"DGE_K{K}_{run_label}_",
                    baseline_contour=contour,
                )

            y_preds[approach].append(y_pred_mean)

            # for plotting calibration and confidence curves later
            if run == 0 and plot:
                y_preds_for_plotting[approach] = y_pred_mean

        # Data aggregated
        X_syn_cat = pd.concat(
            [
                X_syns[i].dataframe()
                for i in range(starting_dataset, starting_dataset + 20)
            ],
            axis=0,
        )
        X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
        X_syn_cat.targettype = X_syns[0].targettype
        X_syn_cat = [X_syn_cat]
        y_pred_mean, _, _ = aggregate(
            X_test,
            X_syn_cat,
            supervised_task,
            models=None,
            workspace_folder=workspace_folder,
            task_type=task_type,
            load=load,
            save=save,
            filename=f"concat_run{run}",
        )

        if include_concat and run == 0 and plot:
            y_preds_for_plotting["DGE$_{20}$ (concat)"] = y_pred_mean

        if plot and d == 2 and run == 0:
            aggregate_imshow(
                X_test,
                X_syn_cat * n_models,
                supervised_task,
                models=None,
                results_folder=results_folder,
                workspace_folder=workspace_folder,
                task_type=task_type,
                load=load,
                save=save,
                filename="concat_all",
                baseline_contour=contour,
            )

        y_preds["DGE$_{20}$ (concat)"].append(y_pred_mean)

    # Evaluation
    # Plotting

    y_true = X_test.dataframe()["target"].values

    if X_syns[0].targettype == "classification" and plot:
        # Consider calibration of different approaches
        fig = plt.figure(figsize=(3, 3), tight_layout=True, dpi=300)
        for key, y_pred in y_preds_for_plotting.items():
            print(key, y_pred.shape)
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            plt.plot(prob_pred, prob_true, label=key)

        plt.xlabel = "Mean predicted probability"
        plt.ylabel = "Fraction of positives"
        plt.tight_layout()
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.legend()

        if save:
            filename = results_folder + "_calibration_curve.png"
            fig.savefig(filename, dpi=300)

        plt.show()
        plt.close()

        plt.figure(figsize=(3, 3), dpi=300)
        for key, y_pred in y_preds_for_plotting.items():
            thresholds, prob_true = accuracy_confidence_curve(y_true, y_pred, n_bins=20)
            plt.plot(thresholds, prob_true, label=key)

        plt.xlabel = r"Confidence threshold \tau"
        plt.ylabel = r"Accuracy on examples \hat{y}"

        plt.legend()
        plt.tight_layout()
        if save:
            filename = results_folder + "_confidence_accuracy_curve.png"
            plt.savefig(filename, dpi=200, bbox_inches="tight")

        if plot:
            plt.show()

        plt.close()

    # Compute metrics

    scores_mean = {}
    scores_std = {}

    scores_all = []
    for approach in y_preds.keys():
        scores = []
        for y_pred in y_preds[approach]:
            scores.append(compute_metrics(y_true, y_pred, X_test.targettype))

        scores = pd.concat(scores, axis=0)
        scores_mean[approach] = np.mean(scores, axis=0)
        scores_std[approach] = np.std(scores, axis=0)
        scores["Approach"] = approach
        scores_all.append(scores)

    scores_all = pd.concat(scores_all, axis=0)
    scores_mean = pd.DataFrame.from_dict(
        scores_mean, orient="index", columns=scores.columns.drop("Approach")
    )
    scores_std = pd.DataFrame.from_dict(
        scores_std, orient="index", columns=scores.columns.drop("Approach")
    )

    return scores_mean, scores_std, scores_all


##############################################################################################################

# Model evaluation and selection experiments


def model_evaluation_experiment(
    X_gt,
    X_syns,
    model_type,
    relative=False,
    workspace_folder=None,
    load=True,
    save=True,
    outlier=False,
    verbose=False,
):
    means = []
    stds = []
    res = {}
    approaches = ["Oracle", "Naive", "DGE$_5$", "DGE$_{10}$", "DGE$_{20}$"]
    K = [None, None, 5, 10, 20]
    if isinstance(outlier, Callable):
        subset = outlier
    elif outlier is True:
        raise ValueError("Subset not properly defined")
    else:
        subset = None
    folder = os.path.join(workspace_folder, "Naive")

    for i, approach in enumerate(approaches):
        if verbose:
            print("Approach: ", approach)
        mean, std, _, all = aggregate_predictive(
            X_gt,
            X_syns,
            models=None,
            task_type=model_type,
            workspace_folder=folder,
            load=load,
            save=save,
            approach=approach,
            relative=relative,
            subset=subset,
            verbose=verbose,
            K=K[i],
        )
        means.append(mean)
        stds.append(std)
        all["Approach"] = approach
        res[approach] = all

    means = pd.concat(means, axis=0)
    stds = pd.concat(stds, axis=0)
    res = pd.concat(res, axis=0)
    if relative == "rmse":
        means = np.sqrt(means)

    means.index = approaches
    stds.index = approaches
    means.index.Name = "Approach"
    stds.index.Name = "Approach"

    return means, stds, res


def model_selection_experiment(
    X_gt,
    X_syns,
    relative="l1",
    workspace_folder="workspace",
    load=True,
    save=True,
    outlier=False,
    model_types=None,
):
    if model_types is None:
        model_types = ["lr", "mlp", "deep_mlp", "rf", "knn", "svm", "xgboost"]

    all_stds = []
    all_means = []
    output_means = {}
    output_stds = {}

    for i, model_type in enumerate(model_types):
        mean, std, _ = model_evaluation_experiment(
            X_gt,
            X_syns,
            model_type,
            workspace_folder=workspace_folder,
            relative=relative,
            load=load,
            save=save,
            outlier=outlier,
        )
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
        sorting = [model_types[i] for i in means.loc["Oracle"].argsort()]
        means_sorted = means.loc[:, sorting]
        stds_sorted = stds.loc[:, sorting]

        for approach in approaches:
            sorting_k = means_sorted.loc[approach].argsort()
            sorting_k = sorting_k.argsort()
            means_sorted.loc[approach + " rank"] = 7 - sorting_k.astype(int)

        output_means[metric] = means_sorted
        output_stds[metric] = stds_sorted

    return output_means, output_stds


def cross_val(
    X_gt,
    X_syns,
    workspace_folder=None,
    results_folder=None,
    save=True,
    load=True,
    task_type="mlp",
    cross_fold=5,
    verbose=False,
):
    """Compares predictions by different approaches using cross validation.

    Args:
        X_test (GenericDataLoader): Test data.
        X_syns (List(GenericDataLoader)): List of synthetic datasets.
        X_test (GenericDataLoader): Real data
        load (bool, optional): Load results, if available. Defaults to True.
        save (bool, optional): Save results when done. Defaults to True.

    Returns:

    """

    if save and results_folder is None:
        raise ValueError("results_folder must be specified when save=True.")

    X_test_r = X_gt.test()

    X_test_r.targettype = X_gt.targettype

    if X_gt.targettype not in ["regression", "classification"]:
        raise ValueError("X_gt.targettype must be regression or classification.")

    # DGE (k=5, 10, 20)
    n_models = 20  # maximum K
    num_runs = len(X_syns) // n_models

    if num_runs > 1 and verbose:
        print("Computing means and stds")

    keys = ["Oracle", "Naive", "DGE$_{20}$", "DGE$_{20}$ (concat)"]
    # keys = keys[-2:]

    # Oracle
    X_oracle = X_gt.train()

    # Oracle ensemble
    scores_r_all = []
    scores_s_all = []

    for run in range(num_runs):
        run_label = f"run_{run}"
        starting_dataset = run * n_models
        scores_s = {}
        scores_r = {}

        for approach in keys:
            kf = KFold(n_splits=cross_fold, shuffle=True, random_state=0)
            print(approach)
            if "oracle" in approach.lower():
                X_syn_run = X_oracle
            elif approach == "Naive":
                X_syn_run = X_syns[run]
            elif approach.startswith("DGE") and "concat" not in approach:
                K = 20
                X_syn_run = X_syns[starting_dataset : starting_dataset + K]
            # This is not used anywhere
            # elif approach == "DGE$_{20}$ (concat)":
            # X_syn_cat = pd.concat(
            #     [
            #         X_syns[i].dataframe()
            #         for i in range(starting_dataset, starting_dataset + 20)
            #     ],
            #     axis=0,
            # )
            else:
                raise ValueError(f"Unknown approach {approach}")

            scores_s[approach] = [0] * cross_fold
            scores_r[approach] = [0] * cross_fold
            for i, (train_index, test_index) in enumerate(kf.split(X_syn_run)):
                if verbose:
                    print("Run", run, "approach", approach, "split", i)

                if isinstance(X_syn_run, List):
                    X_train = cat_dl([X_syn_run[i] for i in train_index])
                    X_test_s = cat_dl([X_syn_run[i] for i in test_index])
                else:
                    if type(X_syn_run) == pd.DataFrame:
                        pass
                    else:
                        X_syn_run = X_syn_run.dataframe()

                    x_train, x_test = (
                        X_syn_run.loc[train_index],
                        X_syn_run.loc[test_index],
                    )
                    X_train = GenericDataLoader(x_train, target_column="target")
                    X_test_s = GenericDataLoader(x_test, target_column="target")

                X_test_s.targettype = X_syns[0].targettype
                X_train.targettype = X_syns[0].targettype

                filename = os.path.join(
                    workspace_folder,
                    f"cross_validation_{task_type}_{approach}_{run_label}_split_{i}.pkl",
                )

                if load and os.path.exists(filename):
                    with open(filename, "rb") as f:
                        model = pickle.load(f)

                elif load and approach == "DGE$_{20}$":
                    # for compatibility with old files
                    alt_filename = os.path.join(
                        workspace_folder,
                        f"cross_validation_{task_type}_"
                        + "DGE$_{20]$"
                        + f"_{run_label}_split_{i}.pkl",
                    )
                    if os.path.exists(alt_filename):
                        with open(alt_filename, "rb") as f:
                            model = pickle.load(f)
                    else:
                        model = None

                else:
                    model = None
                scores_s[approach][i], model = tt_predict_performance(
                    X_test_s,
                    X_train,
                    model=model,
                    model_type=task_type,
                    subset=None,
                    verbose=False,
                )
                scores_r[approach][i], _ = tt_predict_performance(
                    X_test_r,
                    X_train,
                    model=model,
                    model_type=task_type,
                    subset=None,
                    verbose=False,
                )

                scores_s[approach][i]["run"] = run
                scores_r[approach][i]["run"] = run
                scores_s[approach][i]["split"] = i
                scores_r[approach][i]["split"] = i
                scores_s[approach][i]["approach"] = approach
                scores_r[approach][i]["approach"] = approach

                if save and not os.path.exists(filename):
                    with open(filename, "wb") as f:
                        pickle.dump(model, f)

            scores_s[approach] = pd.concat(scores_s[approach], axis=0)
            scores_r[approach] = pd.concat(scores_r[approach], axis=0)

        scores_s_all.append(pd.concat(scores_s))
        scores_r_all.append(pd.concat(scores_r))

    scores_s_all = pd.concat(scores_s_all, axis=0)
    scores_r_all = pd.concat(scores_r_all, axis=0)

    scores_s_mean = scores_s_all.groupby(["run", "approach"]).mean()
    scores_r_mean = scores_r_all.groupby(["run", "approach"]).mean()

    return scores_s_mean, scores_r_mean
