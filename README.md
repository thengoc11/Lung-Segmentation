
# LUNG SEGMENTATION
Welcome to the LUNG SEGMENTATION repository. This project provides an easy-to-use framework of the segmentation model for Medical Image, allowing you to  effectively train/evaluate.

## Requirements
All the dependencies can be installed using the provided requirements.txt file.
## Installation
```bash
1. Clone the repository:
   ```
    git clone https://github.com/thengoc11/Lung-Segmentation
   ```
2. Change the directory:
  ```
    cd Lung-Segmentation
  ```
3. Create a conda environment and install dependencies:

   ```
    conda create -n lung python=3.10
   ```

4. Activate the conda environment:

   ```
     conda activate lung
   ```

## Download LIDC Dataset
```bash
mkdir data
cd data
```

[LIDC data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254&fbclid=IwAR1vDkrpq0IJN8KwPT2Fft1GJ4bFPiMqXp4p08eEfOaUYofS-88pnNF_Z7g)

```bash
# Data Preprocessing with repository
git clone https://github.com/jaeho3690 LIDC-IDRI-Preprocessing
```


## How to run

Train model with default configuration

```bash
# before training, set env WANDB_API_KEY to log with wandb logger
export WANDB_API_KEY = ${WANDB_API_KEY}

# train on one GPU
python src/train.py trainer=gpu logger=wandb

# train on multi GPU
python src/train.py trainer=ddp trainer.devices=4 logger=wandb
