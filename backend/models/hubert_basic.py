# models/hubert_basic.py
import json
import os
from typing import Any, Dict, List

import joblib
import librosa
import numpy as np
import torch
from interfaces import InferenceModel
from transformers import AutoProcessor, HubertModel

DEFAULT_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


class Hubert_basicModel(InferenceModel):
    """
    HuBERT Basic:
    - Uses pretrained HuBERT-base to get embeddings
    - Applies the exported scaler
    - Runs the exported MLP classifier on top
    """

    def __init__(self, export_dir: str):
        super().__init__(export_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        self.max_duration = 3.0  # seconds

        model_name = "facebook/hubert-base-ls960"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.encoder = HubertModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

        # Full classifier
        clf_path = os.path.join(export_dir, "hubert_basic_mlp_full.pt")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(
                f"Cannot find HuBERT Basic classifier at {clf_path}"
            )
        self.classifier = torch.load(clf_path, map_location=self.device)
        self.classifier.to(self.device)
        self.classifier.eval()

        # Scaler
        scaler_path = os.path.join(export_dir, "hubert_basic_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Cannot find scaler at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # Labels
        labels_path = os.path.join(export_dir, "emotion_labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.emotion_labels = json.load(f)
        else:
            self.emotion_labels = DEFAULT_LABELS

    def _load_waveform(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        target_len = int(self.sample_rate * self.max_duration)
        y = librosa.util.fix_length(y, size=target_len)
        return y

    def _embed(self, y: np.ndarray) -> np.ndarray:
        """Compute a single pooled HuBERT embedding (1D)."""
        inputs = self.processor(
            y,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            out = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            hidden = out.last_hidden_state  # (1, T, H)
            pooled = hidden.mean(dim=1)  # (1, H)

        return pooled.cpu().numpy()[0]  # (H,)

    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for path in inputs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

            y = self._load_waveform(path)
            emb = self._embed(y)  # (H,)

            # Scale same way as training
            emb_scaled = self.scaler.transform(emb.reshape(1, -1))  # (1, D)
            x = torch.from_numpy(emb_scaled).float().to(self.device)

            with torch.no_grad():
                logits = self.classifier(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            pct = {
                label: float(p * 100.0) for label, p in zip(self.emotion_labels, probs)
            }
            top_idx = int(np.argmax(probs))
            top_label = self.emotion_labels[top_idx]
            top_score = float(probs[top_idx] * 100.0)

            results.append(
                {
                    "probs": pct,
                    "top_label": top_label,
                    "top_score": top_score,
                }
            )

        return results
