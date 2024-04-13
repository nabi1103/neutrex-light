import argparse
import os
import glob
import torch
from torchinfo import summary
from plot_edc_curve import plot_edc_curve
from plot_kde import plot_kde
from compute_neutrex_vals import compute_neutrex_vals
from compute_score_differential import compute_score_differential

parser = argparse.ArgumentParser(description='Args for the experiment')
parser.add_argument('--expdir', type=str, required = True, help = 'Directory to saved experiment')
parser.add_argument('--expname', required = True, help = 'Name of the experiment')
args = parser.parse_args()

expdir = "/home/stthnguye/neutrex-lite/experiment/" + args.expdir
encoders = glob.glob(expdir + "/*.pth")
custom_cse = None
custom_ee = None
if encoders != None:
    for encoder in encoders:
        if "cse" in encoder.split("/")[-1]:
            custom_cse = torch.load(encoder)
        if "ee" in encoder.split("/")[-1]:
            custom_ee = torch.load(encoder)

for dataset in ["multipie", "feafa"]:
    compute_neutrex_vals(custom_cse, custom_ee, expdir, dataset)
    plot_edc_curve(expdir, args.expname, dataset)
    plot_kde(expdir, args.expname)
    score_diff = str(compute_score_differential(expdir, dataset))
    cse_stats = "Not using custom encoder"
    ee_stats = "Not using custom encoder"

    if custom_cse != None:
        cse_stats = str(summary(custom_cse, (1,3,224,224), verbose=0))

    if custom_ee != None:
        ee_stats = str(summary(custom_ee, (1,3,224,224), verbose=0))

    with open(expdir + "/report_" + args.expdir + "_" + dataset +".txt", "w") as text_file:
        text_file.write("Average Neutrex Score Differential: ")
        text_file.write(score_diff)
        text_file.write("\n")
        text_file.write("Custom Coarse Shape Encoder Summary:\n")
        text_file.write(str(cse_stats))
        text_file.write("\n")
        text_file.write("Custom Expression Encoder Summary:\n")
        text_file.write(str(ee_stats))

print("Benchmark finished")
