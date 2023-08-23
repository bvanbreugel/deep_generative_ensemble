from deep_generative_ensemble.DGE_data import get_real_and_synthetic
from deep_generative_ensemble.DGE_experiments import cross_val
from deep_generative_ensemble.DGE_utils import get_folder_names

num_runs = 10
model_type = "deepish_mlp"
model_name = "ctgan_deep"
nsyn = 5000
max_n = 5000
p_train = 0.8
n_models = 20
cross_fold = 5
load_syn = True
load = True
save = True
verbose = True

scores_s_all = {}
scores_r_all = {}


for dataset in ["moons", "circles", "breast_cancer", "adult", "covid", "seer"]:
    workspace_folder, results_folder = get_folder_names(
        dataset, model_name, max_n=max_n, nsyn=nsyn
    )

    X_gt, X_syns = get_real_and_synthetic(
        dataset=dataset,
        p_train=p_train,
        n_models=n_models * num_runs,
        model_name=model_name,
        load_syn=load_syn,
        verbose=verbose,
        max_n=max_n,
        nsyn=nsyn,
    )

    print(f"Dataset {dataset}\n")

    scores_s, scores_r = cross_val(
        X_gt,
        X_syns,
        workspace_folder=workspace_folder,
        results_folder=results_folder,
        save=save,
        load=load,
        task_type=model_type,
        cross_fold=cross_fold,
        verbose=verbose,
    )

    scores_s_all[dataset] = scores_s
    scores_r_all[dataset] = scores_r
