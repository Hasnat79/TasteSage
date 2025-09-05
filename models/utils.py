import yaml
from types import SimpleNamespace

def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)
