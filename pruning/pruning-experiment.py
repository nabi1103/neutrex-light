import torch
from torchinfo import summary
from nni.compression.pruning import L1NormPruner, L2NormPruner, FPGMPruner
from nni.compression.speedup import ModelSpeedup
from nni.compression.utils import auto_set_denpendency_group_ids
import os

## Load the extracted encoders
# Coarse-shape encoder
cse = torch.load("/home/stthnguye/neutrex-lite/assets/extracted/cse.pth")
# Expression encoder
ee = torch.load("/home/stthnguye/neutrex-lite/assets/extracted/ee.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for encoder in [ee]:
    for s in [0.1, 0.2]:
        config_list = [{
            'op_types': ['Linear', 'Conv2d'],
            'sparse_ratio': s
        }]

        dependency_config = auto_set_denpendency_group_ids(encoder, config_list, torch.rand(1, 3, 224, 224).to(device))

        # Leave the final layer untouched to maintain output shape
        for l in dependency_config:
            if l["op_names"][0] == "layers.2":
                dependency_config.remove(l)

        pruner = FPGMPruner(encoder, dependency_config)
        _, masks = pruner.compress()

        ModelSpeedup(encoder, torch.rand(1, 3, 224, 224).to(device), masks).speedup_model()

        if encoder == cse:
            path = "/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-cse-" + str(s)
            os.mkdir(path)
            path = path + "/pruned-cse.pth"
            torch.save(encoder, "/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-cse-" + str(s) + "/pruned-cse.pth")
        if encoder == ee:
            path = "/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-" + str(s)
            os.mkdir(path)
            path = path + "/pruned-ee.pth"
            torch.save(encoder, "/home/stthnguye/neutrex-lite/experiment/pruning-geometric-global-ee-" + str(s) + "/pruned-ee.pth")
