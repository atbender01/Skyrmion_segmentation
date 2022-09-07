# Skyrmion_segmentation
Skyrmion_segmentation is a codebase for training a neural network and apply already generated nerual networks to apply segmentation to skyrmion lattices. Magnetizations outputs are simualted using micromagnetics, and utilizes the PyLorentz package to create training images from magnetizations. Skyrmion_segmentation includes several features:

- make_training_data
    - run_training_sims.py -- Generates simulated skyrmion data via Mumax, a micromagnetic simulation software package.
    - make_training_from_ovf.ipynb -- Converts simualted data into applicable training data, training images and corresponding ground truths.
- Skyrm_testing_kit
    - Spread of already generated data to test accuracy of neural networks
- trained_SkyrmNet
    - Set of neural networks ready to be used on data
- train_SkyrmNet.ipynb
    - Adds simulated noise to training data. Generates and tests neural networks 

# Getting started
You can clone the repo directly, fork the project, or download the files directly in a .zip. 

Several standard packages are required which can be installed with conda or pip. Environment.yml files are included in the PyLorentz/envs/ folder. Select the appropriate file for your system and create the environment from a command line with 
```
conda env create -f environment.yml
```
Activate with either 
```
source activate Pylorentz
```
or
```
conda activate Pylorentz
```

# PyLorentz
PyLorentz is a codebase designed for analyzing Lorentz Transmission Electron Microscopy (LTEM) data. There are three primary features and functions: 

- PyTIE -- Reconstructing the magnetic induction from LTEM images using the Transport of Intensity Equation (TIE)
- SimLTEM -- Simulating phase shift and LTEM images from a given magnetization 
- GUI -- GUI provided for PyTIE reconstruction.

For full documentation please check outdocumentation pages: [![Documentation Status](https://readthedocs.org/projects/pylorentztem/badge/?version=latest)](https://pylorentztem.readthedocs.io/en/latest/?badge=latest) 

If you use PyLorentz, please cite [paper](https://doi.org/10.1103/PhysRevApplied.15.044025) [1] and this PyLorentz code: [![DOI](https://zenodo.org/badge/263821805.svg)](https://zenodo.org/badge/latestdoi/263821805)

See full PyLorentz README for full documentation and explanation.