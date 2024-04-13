import edc_utils
import numpy as np
import pandas as pd
import os
import glob
from matplotlib import pyplot as plt

def plot_edc_curve(expdir, expname, dataset):
    names = glob.glob(expdir + "/*.csv")
    plot_name = "edc"
    # Original Neutrex score is fixed & Comparison score is fixed
    if dataset == "multipie":
        neutrex_csv = pd.read_csv("../assets/csv/benchmark/neutrex_results_multipie_new.csv", sep=",", header=0)
        comp_scores_csv = pd.read_csv("../assets/csv/benchmark/magface_comparison_scores_multipie_new.csv", sep=",", header=0)
        plot_name = plot_name + "_multipie"
        
    elif dataset == "feafa":
        neutrex_csv = pd.read_csv("../assets/csv/benchmark/neutrex_results_feafa.csv", sep=",", header=0)
        comp_scores_csv = pd.read_csv("../assets/csv/benchmark/magface_comparison_scores_feafa.csv", sep=",", header=0)
        plot_name = plot_name + "_feafa"
    # Merge and sort the fixed scores
    merged = pd.merge(neutrex_csv, comp_scores_csv, on='probe_fnames', how='outer')
    merged.dropna(subset=['neutrex_scores'], inplace=True)
    merged.dropna(subset=['comp_score'], inplace=True)
    neutrex_scores = merged["neutrex_scores"].to_numpy()
    comp_scores = merged["comp_score"].to_numpy()
    # Calculate the edc results for Neutrex
    neutrex_edc_results = edc_utils.compute_edc(error_mode = "FNMR", threshold_quantile = 0.1, quality_scores = neutrex_scores, comparison_scores = comp_scores)
    neutrex_edc_pauc = edc_utils.compute_edc_pauc(neutrex_edc_results, discard_fraction_limit = 0.2)
    
    if names != None:
        for name in names:
            if "multipie" in name.split("/")[-1] and dataset == "multipie":
                experiment_csv = pd.read_csv(name, sep=",", header=0)
            if "feafa" in name.split("/")[-1] and dataset == "feafa":
                experiment_csv = pd.read_csv(name, sep=",", header=0)
        merged_exp = pd.merge(experiment_csv, comp_scores_csv, on='probe_fnames', how='outer')
        merged_exp.dropna(subset=['neutrex_scores'], inplace=True)
        merged_exp.dropna(subset=['comp_score'], inplace=True)
        exp_scores = merged_exp["neutrex_scores"].to_numpy()
        exp_edc_results = edc_utils.compute_edc(error_mode = "FNMR", threshold_quantile = 0.1, quality_scores = exp_scores, comparison_scores = comp_scores)
        exp_edc_pauc = edc_utils.compute_edc_pauc(exp_edc_results, discard_fraction_limit = 0.2)
    else:
        print("No experiement found")

    # plt.plot(neutrex_edc_results['discarded'], neutrex_edc_results['error'], label="NeutrEx", color="darkred", linestyle="--", linewidth=2)
    neutrex_label = "NeutrEx (pAUC: " + str(round(neutrex_edc_pauc*100, 2)) + "%)"
    plt.plot(neutrex_edc_results['discarded'], neutrex_edc_results['error'], label=neutrex_label, color="darkred", linestyle="--", linewidth=2)
    plt.xlim((0, 0.3))
    if exp_edc_results != None and exp_edc_pauc != None:
        # plt.plot(exp_edc_results['discarded'], exp_edc_results['error'], label=expname, color="blue", linestyle="--", linewidth=1)
        expname = expname + " (pAUC: " + str(round(exp_edc_pauc*100, 2)) + "%)"
        plt.plot(exp_edc_results['discarded'], exp_edc_results['error'], label=expname, color="blue", linestyle="--", linewidth=1)
    plt.plot([0.0, 0.1],[0.1, 0.0], color = "black", label = "Theoretical Best")
    plt.xlabel("Discarded Fraction with Lowest Quality")
    plt.ylabel("False Non-Match Rate")
    plt.ylim((0.0, 0.11))
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(expdir, plot_name))
    plt.close()
    print("EDC plot finished")

plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.4", "pruning-fpgm-ee-0.4", "multipie")
plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.4", "pruning-fpgm-ee-0.4", "feafa")

plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.5", "pruning-fpgm-ee-0.5", "multipie")
plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.5", "pruning-fpgm-ee-0.5", "feafa")

plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.6", "pruning-fpgm-ee-0.6", "multipie")
plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.6", "pruning-fpgm-ee-0.6", "feafa")

plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.7", "pruning-fpgm-ee-0.7", "multipie")
plot_edc_curve("/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-0.7", "pruning-fpgm-ee-0.7", "feafa")
