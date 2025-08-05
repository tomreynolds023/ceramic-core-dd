# Ceramic Core Defect Detection

This repository contains the main scripts and programs created during my MECH5485 Professional Project, titled:

### Deep Learning-Based Defect Detection for Ceramic Turbine Blade Cores

The system uses a ResNet18 convolutional neural network to detect defects on ceramic cores used during the investment casting of turbine blades. Once trained, this model was integrated within a web-based dashboard, used to analyse images of newly manufactured cores. 

## Dependencies

To run all developed scripts:
* Download Anaconda.
* Import the required environment.
    * Use `Torch-CUDA-env` if you have a CUDA enabled GPU (Nvidia), and a CUDA driver.
    * Use `Torch-CPU-env` otherwise.

Use of CUDA is highly recommended for running `Model_Training.ipynb` or `Hyperparameter_tuning.py` due to high computational demands. The dashboard can be run on low powered CPUs.

## Image_splitter.ipynb
A jupyter notebook for splitting high resolution images of ceramic cores into small sections suitable for CNN training. 

## Model_Training.ipynb
The main jupyter notebook used for training the ResNet. Due to confidentiality requirements, a training dataset is not included in this repository. Example outputs are however included in the notebook.

As discussed in the project report, multiple training options were tested and are available:
* Training methodology 
    * Train from scratch, transfer learning, fine-tuning
* Image augmentations
    * Rotations, flips, normalisation
* Dataset balancing

## Hyperparameter_tuning.py
Similar to `Model_Training.ipynb` but refactored into functions to allow use of RayTune to optimise model hyperparameters.

Must be run on CUDA.

## Dashboard
A web-based dashboard for efficient visualisation of defects on newly manufactured ceramic cores. 

To run the dashboard enter this into a terminal:
```
python Dashboard/main.py
```

The dashboard will open up a web page running in localhost. From here, images of cores can be uploaded and the locations of defects will be identified. See the project report for more details.
