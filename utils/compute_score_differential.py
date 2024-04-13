import numpy as np
import pandas as pd
import os
import glob

def compute_score_differential(expdir, dataset):
    path = expdir
    names = glob.glob(path + "/*.csv")
    if dataset == "multipie":
        neutrex_csv = pd.read_csv("../assets/csv/benchmark/neutrex_results_multipie_new.csv", sep=",", header=0)
    elif dataset == "feafa":
        neutrex_csv = pd.read_csv("../assets/csv/benchmark/neutrex_results_feafa.csv", sep=",", header=0)
    
    if names != None:
        for name in names:
            if "multipie" in name.split("/")[-1] and dataset == "multipie":
                experiment_csv = pd.read_csv(name, sep=",", header=0)
            if "feafa" in name.split("/")[-1] and dataset == "feafa":
                experiment_csv = pd.read_csv(name, sep=",", header=0)
        merged_exp = pd.merge(neutrex_csv, experiment_csv, on='probe_fnames', how='outer')
        diff = 0
        for index, row in merged_exp.iterrows():
            try:
                diff = diff + abs(row["neutrex_scores_x"] - row["neutrex_scores_y"])
            except:
                print("Missing score at", row )
        return diff / len(merged_exp)
    else:
        print("No experiement found")

