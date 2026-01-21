# utils/seed.py
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Why:
    - Your report/demo should be repeatable.
    - Same seed -> same train/val split and similar metrics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA deterministic (may reduce speed a little)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
