"""AI HIRING GENDER BIAS PROJECT
AUTHOR: HONGYI JI
"""

_version_ = "1.0.0"

from pathlib import Path
import yaml, pydantic,json

class Settings(pydantic.BaseModel):
    sample_size: int = 30000
    random_seed: int = 4 
    mitigation: list[str] = ['reweighing', 'exponentiated_gradient']
    data_dir: Path = Path('data')
    out_dir: Path = Path('outputs')

def load_config(path: Path | None = None) -> Settings:
    cfg_path = Path(path) if path else Path('configs/default.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return Settings(**cfg)



    
