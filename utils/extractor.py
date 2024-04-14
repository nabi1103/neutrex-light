import sys
sys.path.append('../neutrex')
import gdl
import torch
from gdl_apps.EMOCA.utils.load import load_model
from pathlib import Path

# Load EMOCA
emoca, conf = load_model("/home/stthnguye/neutrex/assets/EMOCA/models/", "EMOCA_v2_lr_mse_20", "detail")
emoca.cuda()
emoca.eval()

# Extract the encoders
# coarse_shape_encoder = emoca.deca.E_flame
# torch.save(coarse_shape_encoder, "/home/stthnguye/neutrex-lite/assets/extracted/cse.pth")

# expression_encoder = emoca.deca.E_expression
# torch.save(expression_encoder, "/home/stthnguye/neutrex-lite/assets/extracted/ee.pth")

# detail_encoder = emoca.deca.E_detail
# torch.save(detail_encoder, "/home/stthnguye/neutrex-lite/assets/extracted/de.pth")


