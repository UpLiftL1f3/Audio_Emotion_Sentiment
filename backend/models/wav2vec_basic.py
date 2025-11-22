# models/wav2vec_basic.py

import json
import os
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from interfaces import InferenceModel
from torch import nn
from transformers import AutoProcessor, Wav2Vec2Config, Wav2Vec2Model

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


class Wav2Vec2Classifier(nn.Module):
    """
    Matches the Wav2Vec2Classifier you defined in Wave2Vec_Basic.ipynb:

    - Wav2Vec2Model backbone
    - Mean-pooling over time
    - classifier = Sequential(Linear(H, H), ReLU, Dropout(0.1), Linear(H, num_labels))
    """

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        # Construct Wav2Vec2 backbone from pretrained weights
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.wav2vec2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state  # (B, T, H)

        # Simple mean pooling over time
        pooled = last_hidden_state.mean(dim=1)  # (B, H)

        logits = self.classifier(pooled)  # (B, num_labels)
        return logits


class Wav2vec_basicModel(InferenceModel):
    """
    Wav2Vec2 Basic model used for inference.

    - Reads classifier_config.json for base model name + num_labels.
    - Reconstructs Wav2Vec2Classifier and loads wav2vec_basic_classifier.pt (state_dict).
    - Uses the exported processor for feature extraction.
    - predict(inputs) expects a list of audio file paths.
    """

    def __init__(self, export_dir: str):
        super().__init__(export_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        self.max_duration = 3.0  # seconds

        # --- Load classifier config (model_name, num_labels) ---
        cfg_path = os.path.join(export_dir, "classifier_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing classifier_config.json at {cfg_path}")

        with open(cfg_path, "r") as f:
            clf_cfg = json.load(f)

        model_name = clf_cfg.get("model_name", "facebook/wav2vec2-base")
        num_labels = int(clf_cfg.get("num_labels", 8))

        # --- Load processor from local export dir ---
        self.processor = AutoProcessor.from_pretrained(export_dir)

        # --- Build classifier and load state_dict ---
        self.model = Wav2Vec2Classifier(model_name=model_name, num_labels=num_labels)
        clf_path = os.path.join(export_dir, "wav2vec_basic_classifier.pt")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(
                f"Missing wav2vec_basic_classifier.pt at {clf_path}"
            )

        state_dict = torch.load(clf_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print("Wav2Vec Basic load_state_dict - missing keys:", missing)
        if unexpected:
            print("Wav2Vec Basic load_state_dict - unexpected keys:", unexpected)

        self.model.to(self.device)
        self.model.eval()

        # --- Emotion labels ---
        labels_path = os.path.join(export_dir, "emotion_labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.emotion_labels = json.load(f)
        else:
            self.emotion_labels = DEFAULT_LABELS

    def _load_waveform(self, path: str) -> np.ndarray:
        """Load mono waveform and pad/trim to max_duration seconds."""
        y, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        target_len = int(self.sample_rate * self.max_duration)
        y = librosa.util.fix_length(y, size=target_len)
        return y

    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        inputs: list of file paths to audio files.
        Returns: list of dicts:
          {
            "probs": {label: percentage},
            "top_label": <str>,
            "top_score": <float>
          }
        """
        results: List[Dict[str, Any]] = []

        for path in inputs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

            # 1) Load waveform
            y = self._load_waveform(path)

            # 2) Processor -> input_values tensor
            proc = self.processor(
                y,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = proc["input_values"].to(self.device)
            attention_mask = proc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # 3) Forward pass
            with torch.no_grad():
                logits = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (num_labels,)

            # 4) Convert to percentages + argmax
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
