'''
잡다하게 쓰는 함수는 여기다 모아놓을것

가시성 없는 코드는 죽음을 의미한다...
'''
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path


# 경로에서 타임스탬프 파싱하는 함수
def parse_timestamp(name: str, prefix: str) -> str:
        try:
            return name.split(prefix)[1].split('_')[0]
        except Exception:
            return datetime.now().strftime("%Y%m%d_%H%M%S")

