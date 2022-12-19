# ML4FG RBP Binding Preference Prediction

##Installation
Using conda, install the dependencies:

```
$ conda env create -f environment.yml
$ conda activate rbp
```

Unzip the data in `data.zip`. Depending on the`include_graph` parameter in the config, the processing step will differ.
If it is `True`, the distance matrix will be added to the processed dataset. This will take much longer so it is set to `False` by default.

##Training
To train the model run:

`$ python run_rbp_model.py train configs/config.json`

By default logging is done with Weights & Biases. To use CSV logging run:
`$ python run_rbp_model.py train configs/config.json --no-wandb`