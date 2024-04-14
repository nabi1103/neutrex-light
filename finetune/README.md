Activate the finetune environment to run `finetune.py`

`conda activate finetune`

Replace `neutrex/gdl/datasets/ImageTestDataset.py` with our implementation in `misc/ImageTestDataset.py`. More specifically, the class `CustomTestData` and function `transform_image_tensor` from line 183 to line 350 are new and needs to be added.

The following lines need to be adjusted  if you want to try different finetuning config

<ul>
    <li>Line 16, 18, 20, 22, 24, 150 to set the paths to reelevant resources</li>
    <li>Line 26 to 28 to set training setting</li>
    <li>Line 115 to choose where to save the model</li>
</ul>