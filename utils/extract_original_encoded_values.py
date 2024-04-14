import torch
import os
import sys
import pandas as pd
from tqdm import tqdm, auto
from torchinfo import summary
sys.path.append('../neutrex')
import gdl
from gdl.datasets.ImageTestDataset import TestData
from io_emoca import emoca_encode_flame_no_decompose

## Load the extracted encoders
# Coarse-shape encoder
cse = torch.load("/home/stthnguye/neutrex-lite/assets/extracted/cse.pth")
# Expression encoder
ee = torch.load("/home/stthnguye/neutrex-lite/assets/extracted/ee.pth")

cse.cuda()
ee.cuda()

cse.eval()
ee.eval()

paths = ["/home/stthnguye/dataset/affectnet/images"]
for path in paths:
    dataset = TestData(path, face_detector="fan", max_detection=1)
    subdir_name = "affectnet-reduced"
    os.makedirs("/home/stthnguye/neutrex-lite/assets/finetune/tensors/" + subdir_name, exist_ok = True) 
    img_names = []
    cse_names = []
    ee_names = []
    count = 0
    for img in tqdm(dataset):
        if count == 50000:
            break
        img["image"] = img["image"].cuda()
        if len(img["image"].shape) == 3:
            img["image"] = img["image"].view(1,3,224,224)
        
        image = img["image"]
        
        if len(image.shape) == 5:
                K = image.shape[1]
        elif len(image.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        image = image.view(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        cse_val = cse(image)
        ee_val = ee(image)
        cse_name = "/home/stthnguye/neutrex-lite/assets/finetune/tensors/" + subdir_name + "/" + img['image_name'][0][:-2] + "_cse.pt"
        ee_name = "/home/stthnguye/neutrex-lite/assets/finetune/tensors/" + subdir_name + "/" + img['image_name'][0][:-2] + "_ee.pt"
        torch.save(cse_val, cse_name)
        torch.save(ee_val, ee_name)
        img_names.append(img['image_name'][0][:-2] + ".jpg")
        cse_names.append(cse_name)
        ee_names.append(ee_name)
        count = count + 1

    outpath = "/home/stthnguye/neutrex-lite/assets/finetune/tensors/csv"
    cse_csv = pd.DataFrame({"img_names": img_names, "labels": cse_names})
    ee_csv = pd.DataFrame({"img_names": img_names, "labels": ee_names})
    cse_csv.to_csv(outpath + "/cse_labels_affectnet_50000.csv")
    ee_csv.to_csv(outpath + "/cse_labels_affectnet_50000.csv")
