import os, sys, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "sg_11_2")
OUT_DIR  = os.path.join(BASE_DIR, "novel_results")

SEQ_LEN      = 20
EPOCHS       = 40
BATCH_SIZE   = 32
LR           = 5e-4
HIDDEN       = 64
LAYERS       = 2
DROPOUT      = 0.3
CONV_FILTERS = 64
CONV_KERNEL  = 3
MC_PASSES    = 30
CORR_THRESH  = 0.3
PATIENCE     = 8
TRAIN_FRAC   = 0.8
VAL_FRAC     = 0.1

# mcvar_novel config
TOP_K        = 15
PROFILE      = 'moderate' 
ALPHA_BASE   = 0.95
ENTROPY_REG  = 0.05
RF_ANNUAL    = 0.065
N_DAYS       = 256

MAX_DAILY_RETURN = 0.10
INPUT_COLS       = ['Smooth_Close', 'MA_10', 'EMA_10']

os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
