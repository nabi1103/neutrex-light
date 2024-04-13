import sys
sys.path.append('../neutrex')
import gdl
import torch
from gdl_apps.EMOCA.utils.load import load_model
from pathlib import Path
from torchinfo import summary

cse_path = "/home/stthnguye/neutrex-lite/experiment/combine-model-test/pruned-cse.pth"
ee_path = "/home/stthnguye/neutrex-lite/experiment/combine-model-test/pruned-finetuned-ee.pth"
neutrex_lite_path = "/home/stthnguye/neutrex-lite/experiment/combine-model-test/neutrex-lite.pth"

# Load EMOCA
emoca, conf = load_model("/home/stthnguye/neutrex/assets/EMOCA/models/", "EMOCA_v2_lr_mse_20", "detail")
emoca.cuda()
emoca.eval()

cse = torch.load(cse_path)
ee = torch.load(ee_path)

emoca.deca.E_flame = cse
emoca.deca.E_expression = ee
torch.save(emoca, neutrex_lite_path)
# summary(ee)
