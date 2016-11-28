import os
import sys

import utils_array

import data

import obs_models
import alloc_models
import training_algs

import scripts

import viz

# Bring common functions to root namespace
train_model = scripts.train_model.train_model

bnpy_repo_path, _ = os.path.split(__path__[0])
DATASET_PATH = os.path.join(bnpy_repo_path, 'datasets')
