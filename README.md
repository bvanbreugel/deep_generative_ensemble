# Synthetic data, real errors: how (not) to publish and use synthetic data 

This library implements accompanies the paper:
"Synthetic data, real errors: how (not) to publish and use synthetic data", ICML 2023.
https://arxiv.org/abs/2305.09235

The main focus is the proposed Deep Generative Ensemble (DGE), and its comparison to naive synthetic data methods.

## Installation
Create environment and install packages:
```bash
$ conda create -n synthetic_errors python=3.8
$ conda activate synthetic_errors
$ pip install -r requirements.txt
$ pip install .
```
This code uses the generative modelling library of Synthcity (https://github.com/vanderschaarlab/synthcity)

## Run experiments

All experiments are provided in the notebook main_experiments.ipynb

## Cite
```
@inproceedings{breugel2023synthetic,
  title={Synthetic data, real errors: how (not) to publish and use synthetic data},
  author={van Breugel, Boris and Qian, Zhaozhi and van der Schaar, Mihaela},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```




