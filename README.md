# Liver Tumor Segmentation with Random Shape Pseudo anomaly and two stage training

## Data Download
### Altas
Visit https://www.synapse.org/#!Synapse:syn3193805/wiki/217789, register for the challenge, go to (https://www.synapse.org/#!Synapse:syn3379050) and download the 'RawData.zip'. Unzip the file and put it under `./data/Altas/`.
To preprocess the dataset, run the following command
```
python data_preprocessing.py --phase train
```

### LiTs
Visit LiTs Challenge (https://www.google.com/search?client=safari&rls=en&q=lits+challenge&ie=UTF-8&oe=UTF-8), register for the event, download the dataset and put the unzipped folder under `./data/LiTs`.
To preprocess the dataset, run the following command
```
python data_preprocessing.py --phase tests
```

The preprocessed images would be saved at `./data/liver_dataset`.


## Training
### Stage 1: Reconstruction model
To train the reconstruction model in stage 1, run the following command:
```
python train_reconstruction_stage_1.py
```

After the stage 1 training, the reconstruction model and the corresponding results would be saved at `./outputs`

### Stage 2: Segmentation model
Run
```
python train_segmentation_stage_2.py
```
to implement stage 2 training, the results will be saved under `./outputs`.

## Evaluation
The evaluation is conducted in the training scripts, but if you want to load the pretrained model, run:
```
python train_segmentation_stage_2.py --evaluate True
```

Evaluation results: Sample Auroc0.777, Dice0.556086 (the evaluation results may differ because of the dataset split and model training)


## Results

|           Methods          |  Dice |
|:--------------------------:|:-----:|
|            DRAEM           | 10.55 |
|        Zhang et al.        | 41.08 |
| ASC-Net w/o postprocessing | 32.24 |
|  Asc-Net w postprocessing  | 50.23 |
|            Ours            | 53.03 |