from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

class model:
    def __init__(self):
        self.timestamp = None
        self.args = None
    def train(self, data:pd.DataFrame) -> None:
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