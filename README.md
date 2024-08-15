# NeutrEx-Light: Efficient Expression Neutrality Estimation For Utility Prediction

This repository contains the implementation of <b>NeutrEx-Light: Efficient Expression Neutrality Estimation For Utility Prediction</b>. This work is supervised by Prof. Dr. Christoph Busch and Marcel Grimmer, as part of the master thesis by the same name. A shortened version of this work is accepted for the upcoming [BIOSIG 2024](https://biosig.de/) conference.

Requirements to run the code in this repo.

<ol>
    <li>A Python environment manager, e.g <b>[miniconda](https://docs.anaconda.com/free/miniconda/index.html)</b></li>
    <li>The original [NeutrEx implementation](https://github.com/datasciencegrimmer/neutrex) by Grimmer et al.</li>
</ol>

After installing the NeutrEx repo, put the NeutrEx directory and our repository next to each other in your working folder, i.e

.your/working/folder

├── NeutrEx                      # Original NeutrEx

├── neutrex-lite                 # Our repo

In `env` you can find the environments used in our experiment. It is recommended to install all of them with

`conda env create -f <environment-name>.yml`

Check the respective folders for further infos regarding the experiments

<ol>
    <li><code>utils</code> - benchmarking pipeline for experiments and misc. code</li>
    <li><code>finetune</code> - finetuning experiment</li>
    <li><code>pruning</code> - pruning experiment
</ol>

You can download the experiments reported in the thesis at
[https://drive.google.com/file/d/16UwtRICCzAuVPqNCBGtwy_ZkIFSiikvr/view?usp=sharing](https://drive.google.com/file/d/16UwtRICCzAuVPqNCBGtwy_ZkIFSiikvr/view?usp=sharing)

After downloading, extract the zip file at the root folder of this repo.
