from pathlib import Path
from typing import Mapping

import torch

from .coast import COAST, COASTEmbAbl, COASTwoemb

MODEL_REGISTRY = {
    "COAST": COAST,
    "COASTEmbAbl": COASTEmbAbl,
    "COASTwoemb": COASTwoemb,
}


def create_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](**kwargs)


def load_model_weights(model, path: str | Path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["state_dict"] if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint else checkpoint
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return model.load_state_dict(state, strict=False)


__all__ = [
    "COAST",
    "COASTEmbAbl",
    "COASTwoemb",
    "MODEL_REGISTRY",
    "create_model",
    "load_model_weights",
]
