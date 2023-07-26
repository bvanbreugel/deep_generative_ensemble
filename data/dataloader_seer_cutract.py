# third party
import pandas as pd
import sklearn


def load_seer_cutract(name="seer", reduce_to=20000, seed=42):
    def aggregate_grade(row):
        if row["grade_1.0"] == 1:
            return 1
        if row["grade_2.0"] == 1:
            return 2
        if row["grade_3.0"] == 1:
            return 3
        if row["grade_4.0"] == 1:
            return 4
        if row["grade_5.0"] == 1:
            return 5

    def aggregate_stage(row):
        if row["stage_1"] == 1:
            return 1
        if row["stage_2"] == 1:
            return 2
        if row["stage_3"] == 1:
            return 3
        if row["stage_4"] == 1:
            return 4
        if row["stage_5"] == 1:
            return 5

    def aggregate_treatment(row):
        if row["treatment_CM"] == 1:
            return 1
        if row["treatment_Primary hormone therapy"] == 1:
            return 2
        if row["treatment_Radical Therapy-RDx"] == 1:
            return 3
        if row["treatment_Radical therapy-Sx"] == 1:
            return 4

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
        "stage",
    ]

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment",
        "grade",
        "stage",
    ]

    # features = ['age', 'psa', 'comorbidities', 'treatment_CM', 'treatment_Primary hormone therapy',
    #         'treatment_Radical Therapy-RDx', 'treatment_Radical therapy-Sx', 'grade', 'stage']
    label = "mortCancer"
    df = pd.read_csv(f"./data/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["treatment"] = df.apply(aggregate_treatment, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)
    df["mort"] = df["mort"].astype(int)

    mask = df[label] is True
    df_dead = df[mask]
    df_survive = df[~mask]

    if reduce_to is None:
        n_samples = min([len(df_dead), len(df_survive)])
    else:
        n_samples = reduce_to // 2

    # if n_samples > len(df_dead):
    #     replace_dead = True
    # else:
    #     replace_dead = False

    # if n_samples > len(df_survive):
    #     replace_survive = True
    # else:
    #     replace_survive = False

    # df = pd.concat(
    #     [
    #         df_dead.sample(n_samples, random_state=seed, replace=replace_dead),
    #         df_survive.sample(n_samples, random_state=seed, replace=replace_survive),
    #     ]
    # )

    df = pd.concat(
        [
            df_dead.sample(n_samples, random_state=seed),
            df_survive.sample(n_samples, random_state=seed),
        ]
    )

    df = sklearn.utils.shuffle(df, random_state=seed)
    df = df.reset_index(drop=True)
    return df[features], df[label]
