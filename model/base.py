from datetime import datetime
from pathlib import Path
import random

import pandas as pd
from omegaconf import OmegaConf
import numpy as np

try:
    import torch
except Exception:
    torch = None

class model:
    def __init__(self):
        self.timestamp = None
        self.args = None
    def train(self, data_train:pd.DataFrame, data_valid:pd.DataFrame = None, **kwargs) -> None:
        pass
    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        # 귀찮은데 inference 기능은 여기 통합하는걸로
        pass
    def load(self, model_path:str) -> None:
        pass


    
    def save_args(self, args: dict, args_path: str) -> None:
        if args is None:
                return
        OmegaConf.save(OmegaConf.create(args), args_path)

    def load_args(self, args_path: str) -> dict:
        cfg = OmegaConf.load(args_path)
        self.args = OmegaConf.to_container(cfg, resolve=True)
        return self.args

    def set_seed(self, seed: int) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
