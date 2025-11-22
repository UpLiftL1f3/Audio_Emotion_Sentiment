import importlib
import json
import os
from typing import Any, Dict

from interfaces import InferenceModel


class ModelRegistry:
    def __init__(self, config_path: str = "models.json"):
        self.cfg_path = config_path
        with open(self.cfg_path, "r") as f:
            self.cfg = json.load(f)
        self.cache: Dict[str, InferenceModel] = {}

    def list_models(self):
        return list(self.cfg.keys())

    def get(self, name: str) -> InferenceModel:
        if name in self.cache:
            return self.cache[name]

        if name not in self.cfg:
            raise KeyError(f"Unknown model '{name}'. Known: {self.list_models()}")

        entry = self.cfg[name]
        impl_module = entry["impl"]  # e.g., "multihead"
        export_dir = entry["export_dir"]  # e.g., "export_multihead"

        module = importlib.import_module(f"models.{impl_module}")
        # Convention: class is <ImplName capitalized>Model, e.g., MultiheadModel
        cls_name = f"{impl_module.capitalize()}Model"
        cls = getattr(module, cls_name)
        model: InferenceModel = cls(export_dir=export_dir)
        self.cache[name] = model
        return model
