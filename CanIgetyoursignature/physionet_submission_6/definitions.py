import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set true if submission
SUBMISSION = True


# Setup paths
if ROOT_DIR == '/Users/jambo/Documents/PhD/code/actual/sepsis':
    DATA_DIR = ROOT_DIR + '/data/test'
    MODELS_DIR = ROOT_DIR + '/models/test'
else:
    DATA_DIR = '/scratch/morrill/physionet2019/data'
    MODELS_DIR = ROOT_DIR + '/models'


# Packages/functions used everywhere
from src.omni.decorators import *
from src.omni.functions import *
try:
    from src.omni.base import *
except:
    pass