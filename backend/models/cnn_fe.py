# models/cnn_fe.py
import json
import os
from typing import Any, Dict, List

import librosa
import numpy as np
import tensorflow as tf
from interfaces import InferenceModel


class Cnn_feModel(InferenceModel):
    """
    Feature-Engineered CNN emotion model.
    Expects an exported Keras model and emotion_labels.json in `export_dir`.
    Inputs to `predict` are paths to audio files (.wav).
    """

    def __init__(self, export_dir: str):
        super().__init__(export_dir)

        # --- Load model ---
        model_path = os.path.join(export_dir, "cnn_fe.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find FE CNN model at {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # --- Load labels ---
        labels_path = os.path.join(export_dir, "emotion_labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Cannot find emotion_labels.json at {labels_path}")
        with open(labels_path, "r") as f:
            self.emotion_labels = json.load(f)

        # Optional: per-channel mean/std if you exported them
        mean_path = os.path.join(export_dir, "cnn_fe_feat_mean.npy")
        std_path = os.path.join(export_dir, "cnn_fe_feat_std.npy")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.feat_mean = np.load(mean_path)
            self.feat_std = np.load(std_path)
        else:
            self.feat_mean = None
            self.feat_std = None

        # These must match your FE CNN training notebook
        self.sample_rate = 16000
        self.max_duration = 3.0  # seconds
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 64

    def _compute_mel_stack(self, file_path: str) -> np.ndarray:
        """Compute [log-Mel, delta, delta–delta] stack: (n_mels, T, 3)."""
        y, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)

        target_len = int(self.sample_rate * self.max_duration)
        y = librosa.util.fix_length(y, size=target_len)

        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # delta and delta–delta
        delta = librosa.feature.delta(S_db, order=1)
        delta2 = librosa.feature.delta(S_db, order=2)

        feat = np.stack([S_db, delta, delta2], axis=-1)  # (n_mels, T, 3)

        if self.feat_mean is not None and self.feat_std is not None:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)

        return feat

    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for path in inputs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

            feat = self._compute_mel_stack(path)  # (n_mels, T, 3)
            x = np.expand_dims(feat, axis=0)  # (1, n_mels, T, 3)

            probs = self.model.predict(x, verbose=0)[0]
            probs = probs.astype(float)

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
