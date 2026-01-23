from ..base import model
import pandas as pd
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

from utils import parse_timestamp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

'''
class model:
    def __init__(self):
        pass
    def train(self, data:pd.DataFrame) -> None:
        pass
    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        pass
    def load(self, model_path:str) -> None:
        pass
'''
class mlp(model):
    def __init__(self):
        pass
    def load(self, model_path: str) -> None:
        self.model = torch.load(model_path)
        self.model.eval()
    def train(self):pass

    def predict(self):pass


class MLP_model(nn.Module):
    def __init__(self): pass
        
    def forward(self, x): pass

class dataset(torch.utils.data.Dataset):
    def __init__(self): pass

    def __len__(self): pass

    def __getitem__(self, idx): pass

