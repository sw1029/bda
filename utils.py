'''
잡다하게 쓰는 함수는 여기다 모아놓을것

가시성 없는 코드는 죽음을 의미한다...
'''
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import random
import numpy as np
try:
    import torch
except Exception:
    torch = None

from contextlib import contextmanager


# 경로에서 타임스탬프 파싱하는 함수
def parse_timestamp(name: str, prefix: str) -> str:
        try:
            return name.split(prefix)[1].split('_')[0]
        except Exception:
            return datetime.now().strftime("%Y%m%d_%H%M%S")


# 시드 세팅용
@contextmanager
def set_seed(seed: int):
    if seed is None:
        # No-op context manager when no seed is provided.
        yield
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


    yield



    # 시드 복원
    random.seed()
    np.random.seed()
    if torch is not None:
        torch.manual_seed(torch.initial_seed())
        torch.cuda.manual_seed_all(torch.initial_seed())
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print(f"seed {seed} 에서의 실험 완료.")
