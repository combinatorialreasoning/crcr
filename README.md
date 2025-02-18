# Contrastive Learning for Combinatorial Reasoning
This is the code used for training and evaluation in the paper Contrastive Representations for Combinatorial Reasoning.

## Install requirements
This was tested with python 3.10.
Install the requirements by running `pip install -r requirements.txt`.

## Download or generate the dataset
For the Sokoban dataset, download the dataset and eval boards from the [https://drive.google.com/drive/folders/1vdadRzrlxbHs_O6BkznCmrUehDkzgsOo?usp=sharing]{drive}:

The datasets for other environments are generated:
For the Rubik's cube run:
`python data_generate_rubik.py`

For the NPuzzle run:
`python data_generate_npuzzle.py`


For the Lights Out run:
`python data_generate_lights_out.py`


## Run the training
The configs for experiments are in the files in the folder experiments.

`xxx.py` is for training and solving the problem xxx, 
for `'new_array_dataset.GenDataset.double_batch':` with value 1, contrastive baseline is trained.
For `'new_array_dataset.GenDataset.double_batch':` with value 2, CR^2 is trained.

`xxx_baseline.py` is for training the supervised baseline for solving the problem xxx.
for `'new_array_dataset.GenDataset.double_batch':`  with value 1, contrastive baseline is trained.
For `'new_array_dataset.GenDataset.double_batch':` with value 2, CR^2 is trained.

Moreover, you need to overwrite the path to the dataset:
`'new_array_dataset<_baseline>.GenDataset.path':` for npuzzle, rubik and lights out

`'jobs.generating_job.TrainJob.test_path':` for the test dataset in sokoban

`'new_array_dataset<_baseline>_len.GenDataset.path':` for the training dataset in sokoban
