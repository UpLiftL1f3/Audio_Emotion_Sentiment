"""HuBERT-based emotion classifier: basic variant."""

import json
import os
from typing import Any, Dict, List

import joblib
import librosa
import numpy as np
import torch
from interfaces import InferenceModel
from transformers import AutoFeatureExtractor, HubertModel

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
        # HuBERT-base exposes a feature extractor (no tokenizer).
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.encoder = HubertModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

        # --- scaler ---
        scaler_path = os.path.join(export_dir, "hubert_basic_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # --- labels ---
        labels_path = os.path.join(export_dir, "emotion_labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.emotion_labels = json.load(f)
        else:
            self.emotion_labels = DEFAULT_LABELS

        self.num_labels = len(self.emotion_labels)

        # --- classifier (robust loader) ---
        clf_path = os.path.join(export_dir, "hubert_basic_mlp.pt")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Missing classifier weights at {clf_path}")

        classifier = None
        classifier_type = "torch"

        # Try sklearn/joblib first, then fall back to torch/state-dict.
        try:
            candidate = joblib.load(clf_path)
            if hasattr(candidate, "predict_proba"):
                classifier = candidate
                classifier_type = "sklearn"
        except Exception:
            classifier = None

        self.W = None  # type: ignore[attr-defined]
        self.b = None  # type: ignore[attr-defined]

        if classifier is None:
            state = torch.load(clf_path, map_location="cpu")

            # If it's a full PyTorch module, use it directly.
            if hasattr(state, "eval") and callable(getattr(state, "eval")):
                state.to(self.device)
                state.eval()
                classifier = state
                classifier_type = "torch"
            elif isinstance(state, dict):
                # Assume a single linear layer state_dict: use first 2D weight and 1D bias.
                W = None
                b = None
                for _, v in state.items():
                    if not hasattr(v, "dim"):
                        continue
                    if v.dim() == 2 and W is None:
                        W = v.cpu().numpy()
                    elif v.dim() == 1 and b is None:
                        b = v.cpu().numpy()
                    if W is not None and b is not None:
                        break

                if W is None or b is None:
                    raise TypeError(
                        "Unsupported classifier state_dict format in "
                        "'hubert_basic_mlp.pt' (could not find linear weights/bias)."
                    )

                self.W = W  # (out_dim, in_dim)
                self.b = b  # (out_dim,)
                classifier_type = "linear_state_dict"
                classifier = None
            else:
                raise TypeError(
                    "Unsupported classifier object loaded from 'hubert_basic_mlp.pt'."
                )

        self.classifier = classifier
        self.classifier_type = classifier_type

    def _load_waveform(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        y, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        target_len = int(self.sample_rate * self.max_duration)
        y = librosa.util.fix_length(y, size=target_len)
        return y

    def _embed(self, y: np.ndarray) -> np.ndarray:
        """Compute a single pooled HuBERT embedding (1D)."""
        inputs = self.feature_extractor(
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

    def _encode(self, path: str) -> np.ndarray:
        """Waveform -> HuBERT embedding -> scaled feature vector used by the classifier."""
        y = self._load_waveform(path)
        emb = self._embed(y)  # (H,)
        scaled = self.scaler.transform(emb.reshape(1, -1))  # (1, D)
        return scaled[0]  # (D,)

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, D) feature matrix.
        Returns probs: (N, num_labels).
        """
        if self.classifier_type == "sklearn":
            probs = self.classifier.predict_proba(X)  # type: ignore[operator]
            return probs.astype(float)

        if self.classifier_type == "linear_state_dict":
            assert self.W is not None and self.b is not None
            logits = X @ self.W.T + self.b  # (N, out_dim)
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            return probs.astype(float)

        # Torch classifier module
        with torch.no_grad():
            logits = self.classifier(torch.from_numpy(X).float().to(self.device))  # type: ignore[operator]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs.astype(float)

    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        inputs: list of audio file paths.
        Returns: list of dicts with:
          - "probs": {emotion: percentage_0_100}
          - "top_label": best emotion
          - "top_score": best percentage
        """
        # 1) Encode each path into the same feature space used in training
        feats: List[np.ndarray] = [self._encode(p) for p in inputs]
        X = np.stack(feats, axis=0)  # (N, D)

        # 2) Get raw probs/logits from classifier
        probs_batch = self._predict_batch(X)  # (N, num_outputs)

        results: List[Dict[str, Any]] = []
        for probs in probs_batch:
            probs = np.asarray(probs, dtype=float)

            # Limit to the number of emotion labels we actually have
            n = min(len(self.emotion_labels), len(probs))
            if n == 0:
                continue

            probs_use = probs[:n]
            labels_use = self.emotion_labels[:n]

            # --- Re-normalize over these n emotions so they form a proper distribution ---
            total = float(np.sum(probs_use))
            if (not np.isfinite(total)) or total <= 0.0:
                # Degenerate case: fall back to uniform
                probs_use = np.ones_like(probs_use, dtype=float) / float(n)
            else:
                probs_use = probs_use / total  # now sums to ~1.0

            # Convert to percentages
            pct = {label: float(p * 100.0) for label, p in zip(labels_use, probs_use)}

            top_idx = int(np.argmax(probs_use))
            top_label = labels_use[top_idx]
            top_score = float(probs_use[top_idx] * 100.0)

            results.append(
                {
                    "probs": pct,
                    "top_label": top_label,
                    "top_score": top_score,
                }
            )

        return results
