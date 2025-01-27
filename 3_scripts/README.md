# Scripts

In this section, you will find several scripts to: 

- `train.py`: use to train and save a model built with our training dataset

## Data download and clean

These scripts assume that the data is downloaded, cleaned and available at a certain filepath ([data/clean_fer_2013](../data/clean_fer_2013/)). Downloading and cleaning the data were completed during EDA.

If you haven't already, run the following jupyter notebooks from the root of this project:

- [01_exploratory_data_analysis](../1_eda/01_exploratory_data_analysis.ipynb): run this workbook from the **root** of this project to download the data from Kaggle to `data/fer_2013`. 
- [02_preprocessing](../1_eda/02_preprocessing.ipynb): run this workbook from the **root** of this project to clean the data and save the cleaned data to `data/clean_fer_2013`.

The scripts will assume that the cleaned dataset is available at [data/clean_fer_2013](../data/clean_fer_2013/)


## Environment setup

To run these scripts, make sure that you have set up and activated the Pipenv environment. See [environment_setup.md](../environment_setup.md) for a more detailed description.

Run the following in the root folder of this project:

```bash
pipenv shell
```

## Train xgboost 
TODO: description

To recreate my final model with the parameters that I found produced the best predictive results, run the following **within this directory**

```bash
pipenv shell
python train.py
```
If you want to run this script from the root directory:

```bash
 python 3_scripts/train.py
 ```

The **default parameters are the final tuned parameters**:

- `learning_rate` = 0.001
- `dense_layer_units` = [64, 32]
- `drop_rate` = 0.0
- `n_epochs` = 100
- `use_class_weight` = True
- `use_data_augmentation` = True


### Parameterization (optional)
If you want to train, validate and save a **new** model, the script allows you to do this.

The following parameters can be passed into the script if you want to create a new model.

- `learning_rate_final`: This the learning rate to pass to the Optimizer (Adam). It should be between 0 and 1.
- `dense_layer_units`: This is a list of the number of units for each Dense Layer e.g. `128,64` will create a model with the first dense layer having 128 units and the second having 64". If you do not want to create any new additional layers than use `0`
- `drop_rate`: This is a regularization parameter: the percentage of neurons in a layer to temporarily make inactive. It should be a float between [0,1).
- `n_epochs`: The number of epochs to train the model
- `use_class_weight`: If this is true then the loss function will be modified by the class weights. We have an imbalanced dataset so this is set to True by default. 
- `use_data_augmentation`: If this is true then all the training images will be augmented.

Note that arguments are not validated beyond some basic type checking: it is up to the user to pass in a sensible value. 

You can also see the configurable parameters by running help:

**help**

```bash
python train.py --help 
```

**example script**

Notice to specify that you don't want to use class weights you should run the script with `--no-use_class_weight`. By default class weights will be used.

Similarly, if you don't want to use data augmention, you should run the script with `--no-use_data_augmentation`. By default data augmentation will be used (which is more time consuming).

```bash
python train.py --learning_rate=0.0001 --dense_layer_units=128,64,32 --drop_rate=0.8 --n_epochs=30 --no-use_class_weight --no-use_data_augmentation
```

If you do not want to add any additional dense layers and stick to having one single output layer then pass in `--dense_layer_units=0`

```bash
python train.py --learning_rate=0.0001 --dense_layer_units=0 --drop_rate=0.8 --n_epochs=30 --no-use_class_weight --no-use_data_augmentation
```

