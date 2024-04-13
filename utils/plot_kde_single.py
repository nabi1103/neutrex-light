import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def derive_exp(value):
    return str(value).split("_")[-1]

def plot_kde(expdir, expname, plot_name):
    names = glob.glob(expdir + "/*.csv")
    if names != None:
        for name in names:
            try:
                experiment_csv = pd.read_csv(name, sep=",", header=0)
                experiment_csv.dropna(subset=['probe_fnames'], inplace=True)
                experiment_csv["Expression"] = experiment_csv["probe_fnames"].apply(derive_exp)
                plot = sns.kdeplot(
                    data=experiment_csv,
                    x="neutrex_scores",
                    hue="Expression",
                    fill=True,
                    common_norm=False,
                    palette=sns.color_palette(n_colors=6),
                    alpha=.5,
                    linewidth=0,
                    )
                plot.set(xlim=(0, 105))
                plot.set(ylim=(0, 0.15))
                plot.set_title(expname)
                plot.set_xlabel("NeutrEx Scores")
                sns.move_legend(plot, "upper left")
                if "multipie" in name.split("/")[-1]:
                    plt.savefig(os.path.join(expdir, plot_name + "_multipie"))
                    # plt.savefig(os.path.join(expdir, plot_name + "_multipie_ee03_afnt"))
                if "feafa" in name.split("/")[-1]:
                    plt.savefig(os.path.join(expdir, plot_name + "_feafa"))
                    # plt.savefig(os.path.join(expdir, plot_name + "_feafa_ee03_afnt"))
                plt.close()
            except:
                print(name)
    else:
        print("No experiement found")

plot_kde("/home/stthnguye/neutrex-lite/experiment/neutrex-lite-a", 'neutrex-lite-a', "neutrex-lite-a")

plot_kde("/home/stthnguye/neutrex-lite/experiment/neutrex-lite-b", 'neutrex-lite-b', "neutrex-lite-b")

plot_kde("/home/stthnguye/neutrex-lite/experiment/neutrex-lite-c", 'neutrex-lite-c', "neutrex-lite-c")

plot_kde("/home/stthnguye/neutrex-lite/experiment/neutrex-lite-d", 'neutrex-lite-d', "neutrex-lite-d")
