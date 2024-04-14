from io_emoca import custom_encode_values
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('../neutrex')
import gdl
from torchinfo import summary
from gdl.datasets.ImageTestDataset import TestData
from gdl_apps.EMOCA.utils.load import load_model
from gdl_apps.EMOCA.utils.io import generic_shape_test_custom
from tqdm import tqdm, auto


def compute_neutrex_vals(custom_cse, custom_ee, expdir, dataset):
    # Load EMOCA for rendering
    emoca, conf = load_model("/home/stthnguye/neutrex/assets/EMOCA/models/", "EMOCA_v2_lr_mse_20", "detail")
    emoca.cuda()
    emoca.eval()

    ## Load the extracted encoders
    # Coarse-shape encoder
    if custom_cse != None:
        cse = custom_cse
    else:
        cse = torch.load("/home/stthnguye/neutrex-lite/assets/extracted/cse.pth")
    # Expression encoder
    if custom_ee != None:
        ee = custom_ee
    else:
        ee = torch.load("/home/stthnguye/neutrex-lite/assets/extracted/ee.pth")
    # Detail extractor should not be touched
    de = torch.load("/home/stthnguye/neutrex-lite/assets/extracted/de.pth")

    cse.cuda()
    ee.cuda()
    de.cuda()

    cse.eval()
    ee.eval()
    de.eval()

    if dataset == "multipie":
        dataset = TestData("/home/stthnguye/dataset/MultiPie", face_detector="fan", max_detection=1)
        neut_anchor_verts = np.load("/home/stthnguye/neutrex-lite/assets/neutral-anchor/multipie-neutral-ref-verts.npy")
        outpath = expdir + "/neutrex_results_multipie.csv"

    elif dataset == "feafa":
        dataset = TestData("/home/stthnguye/dataset/FEAFA+", face_detector="fan", max_detection=1)
        neut_anchor_verts = np.load("/home/stthnguye/neutrex-lite/assets/neutral-anchor/feafa+-neutral-ref-verts.npy")
        outpath = expdir + "/neutrex_results_feafa.csv"

    else:
        print("Invalid dataset")
        return
    neutrex_vals = []
    img_names = []

    neut_anchor_verts = neut_anchor_verts.reshape(len(neut_anchor_verts) // 3, 3)

    D_MIN = 0.00045388718717731535
    D_MAX = 0.008605152368545532

    for img in tqdm(dataset):
        # load reconstruct input img and get vertices
        c_vals = custom_encode_values(img, cse, ee, de)
        input_verts, _ = generic_shape_test_custom(emoca, img, c_vals)
        input_verts = input_verts['verts'][0].cpu().numpy()
        # compute neutrex vals (euclidean dists --> min-max-scaling)
        eucl_dist = np.linalg.norm(neut_anchor_verts - input_verts, ord = 2, axis = 1)
        avg_eucl_dist = np.mean(eucl_dist)
        neutrex_val = (avg_eucl_dist - D_MIN) / (D_MAX - D_MIN)
        neutrex_val = 100 * (1 - np.clip(neutrex_val, 0, 1))
        # Store vals and img names
        neutrex_vals.append(neutrex_val)
        img_names.append(img['image_name'][0][:-2])
        
    results = pd.DataFrame({"probe_fnames": img_names, "neutrex_scores": neutrex_vals})
    results.to_csv(outpath)
    print("Neutrex computation finished")
