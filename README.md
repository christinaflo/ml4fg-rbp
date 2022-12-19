# ML4FG RBP Binding Preference Prediction

##Installation
Using conda, install the dependencies:

```
$ conda env create -f environment.yml
$ conda activate rbp
```

Download the data at: https://drive.google.com/file/d/1w--8X0APEYfg-tAHOdH68COm_VOykC6n/view?usp=sharing
Unzip it in the top level directly. Depending on the`include_graph` parameter in the config, the processing step will differ.
If it is `True`, the distance matrix will be added to the processed dataset. This will take much longer so it is set to `False` by default.

##Training
To train the model run:

`$ python run_rbp_model.py train configs/config.json`

By default logging is done with Weights & Biases. To use CSV logging run:
`$ python run_rbp_model.py train configs/config.json --no-wandb`
