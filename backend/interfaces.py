# interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class InferenceModel(ABC):
    """
    Generic interface for inference models.
    For this project, `inputs` will usually be file paths to audio clips.
    Each model returns one dict per input (e.g., probabilities, top label, etc.).
    """

    def __init__(self, export_dir: str):
        self.export_dir = export_dir

    @abstractmethod
    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of inputs (audio file paths for our use case).
        Returns a list of dicts, one per input.
        """
        ...
