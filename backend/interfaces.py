from typing import Any, Dict, List, Protocol


class InferenceModel(Protocol):
    name: str  # machine-readable key, e.g. "multihead"

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return one dict per input text. Shape is model-specific but stable."""
        ...
