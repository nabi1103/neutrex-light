Activate the pruning environment to run `pruning_experiment.py`

`conda activate prune`

The following lines need to be adjusted  if you want to try different pruning config

<ul>
    <li>Line 17 to choose which encoder to prune</li>
    <li>Line 19 to choose which sparse ratio to prune. Valid ratio is between [0.0, 1.0]</li>
    <li>Line 34 to choose which pruner to use</li>
</ul>