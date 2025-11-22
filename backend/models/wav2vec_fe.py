# models/wav2vec_fe.py

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


def compute_prosody_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Match the prosody features from Wave2Vec_Feature.ipynb:

    - f0 mean, f0 std (YIN pitch)
    - RMS energy mean, RMS energy std
    """
    # Pitch using YIN
    f0 = librosa.yin(y, fmin=50, fmax=400)
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        f0_mean, f0_std = 0.0, 0.0
    else:
        f0_mean = float(np.mean(f0))
        f0_std = float(np.std(f0))

    # Energy using RMS
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    return np.array([f0_mean, f0_std, energy_mean, energy_std], dtype=np.float32)


class Wav2Vec2ClassifierFE(nn.Module):
    """
    Matches the Wav2Vec2Classifier in Wave2Vec_Feature.ipynb:

    - Wav2Vec2 backbone
    - Attention pooling over time
    - Concatenate 4-dim prosody features
    - classifier = Sequential(Linear(H+4, 512), ReLU, Dropout(0.3), Linear(512, num_labels))
    """

    def __init__(self, model_name: str, num_labels: int, prosody_dim: int = 4):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.wav2vec2.config.hidden_size

        self.attention = nn.Linear(hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + prosody_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_values, prosody_features=None, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state  # (B, T, H)

        # Attention scores over time
        attn_scores = self.attention(last_hidden_state).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        pooled = (last_hidden_state * attn_weights).sum(dim=1)  # (B, H)

        # Concatenate prosody features if provided
        if prosody_features is not None:
            pooled = torch.cat(
                [pooled, prosody_features], dim=-1
            )  # (B, H + prosody_dim)

        logits = self.classifier(pooled)  # (B, num_labels)
        return logits


class Wav2vec_feModel(InferenceModel):
    """
    Feature-Engineered Wav2Vec2 model:

    - Uses attention pooling + prosody features
    - Loads state_dict from wav2vec_fe_classifier.pt
    - Uses the same processor and labels layout as basic.
    """

    def __init__(self, export_dir: str):
        super().__init__(export_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        self.max_duration = 3.0  # seconds

        # --- Load classifier config ---
        cfg_path = os.path.join(export_dir, "classifier_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing classifier_config.json at {cfg_path}")

        with open(cfg_path, "r") as f:
            clf_cfg = json.load(f)

        model_name = clf_cfg.get("model_name", "facebook/wav2vec2-base")
        num_labels = int(clf_cfg.get("num_labels", 8))
        prosody_dim = int(clf_cfg.get("prosody_dim", 4))

        # --- Processor from local export dir ---
        self.processor = AutoProcessor.from_pretrained(export_dir)

        # --- Build FE classifier and load state_dict ---
        self.model = Wav2Vec2ClassifierFE(
            model_name=model_name,
            num_labels=num_labels,
            prosody_dim=prosody_dim,
        )
        clf_path = os.path.join(export_dir, "wav2vec_fe_classifier.pt")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Missing wav2vec_fe_classifier.pt at {clf_path}")

        state_dict = torch.load(clf_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print("Wav2Vec FE load_state_dict - missing keys:", missing)
        if unexpected:
            print("Wav2Vec FE load_state_dict - unexpected keys:", unexpected)

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
        Returns: list of dicts with probabilities and top prediction.
        """
        results: List[Dict[str, Any]] = []

        for path in inputs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

            # 1) Load waveform
            y = self._load_waveform(path)

            # 2) Compute prosody features (shape: (1, 4))
            prosody_np = compute_prosody_features(y, self.sample_rate)
            prosody_tensor = torch.from_numpy(prosody_np).unsqueeze(0).to(self.device)

            # 3) Processor -> input_values tensor
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

            # 4) Forward pass
            with torch.no_grad():
                logits = self.model(
                    input_values=input_values,
                    prosody_features=prosody_tensor,
                    attention_mask=attention_mask,
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # 5) Convert to percentages + argmax
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
