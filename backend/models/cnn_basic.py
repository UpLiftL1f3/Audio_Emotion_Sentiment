# models/cnn_basic.py
import json
import os
from typing import Any, Dict, List

import librosa
import numpy as np
import tensorflow as tf
from interfaces import InferenceModel


class Cnn_basicModel(InferenceModel):
    """
    Basic CNN emotion model.
    Expects an exported Keras model and emotion_labels.json in `export_dir`.
    Inputs to `predict` are paths to audio files (.wav/.mov after audio extraction).
    """

    def __init__(self, export_dir: str):
        super().__init__(export_dir)

        # --- Load model ---
        model_path = os.path.join(export_dir, "cnn_basic.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find CNN basic model at {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Read expected input shape from the model, e.g. (None, 64, 94, 1)
        input_shape = self.model.input_shape
        # Some Keras models return a list of input shapes; handle that case
        if isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            input_shape = input_shape[0]
        _, self.in_n_mels, self.in_n_frames, _ = input_shape

        # --- Load labels ---
        labels_path = os.path.join(export_dir, "emotion_labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Cannot find emotion_labels.json at {labels_path}")
        with open(labels_path, "r") as f:
            self.emotion_labels = json.load(f)

        # --- Optional: spec mean/std exported from Colab ---
        mean_path = os.path.join(export_dir, "cnn_basic_spec_mean.npy")
        std_path = os.path.join(export_dir, "cnn_basic_spec_std.npy")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.spec_mean = np.load(mean_path)
            self.spec_std = np.load(std_path)
        else:
            self.spec_mean = None
            self.spec_std = None

        # These must match the training notebook
        self.sample_rate = 16000  # SAME as in your CNN notebook
        self.max_duration = 3.0  # seconds
        self.n_fft = 1024  # adjust if different in your notebook
        self.hop_length = 512  # adjust if different
        self.n_mels = 64  # adjust if different

    # ---- internal helper ----

    def _compute_log_mel(self, file_path: str) -> np.ndarray:
        """Recreate the log-Mel spectrogram exactly like in the training notebook."""
        y, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)

        target_len = int(self.sample_rate * self.max_duration)
        y = librosa.util.fix_length(y, size=target_len)

        S = librosa.feature.melspectrogram(
            y=y,  # <-- IMPORTANT: keyword, not positional
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Make sure the time dimension (number of frames) matches what the CNN expects.
        # The model input shape is (None, in_n_mels, in_n_frames, 1).
        # Our spectrogram is (n_mels, T). We will center-crop or pad along T to match.
        T = S_db.shape[1]
        target_T = int(self.in_n_frames)
        if T > target_T:
            # Center-crop in time
            start = (T - target_T) // 2
            S_db = S_db[:, start : start + target_T]
        elif T < target_T:
            # Pad with zeros (silence) to reach the required number of frames
            pad_left = (target_T - T) // 2
            pad_right = target_T - T - pad_left
            S_db = np.pad(
                S_db,
                pad_width=((0, 0), (pad_left, pad_right)),
                mode="constant",
                constant_values=0.0,
            )

        # Optional normalization
        if self.spec_mean is not None and self.spec_std is not None:
            S_db = (S_db - self.spec_mean) / (self.spec_std + 1e-8)

        return S_db  # shape: (n_mels, T)

    # ---- required interface ----
    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        `inputs` is a list of audio file paths.
        Returns one dict per file with:
          - "probs": {emotion: percentage_0_100}
          - "top_label": best emotion
          - "top_score": best percentage
        """
        results: List[Dict[str, Any]] = []

        for path in inputs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

            spec = self._compute_log_mel(path)  # (n_mels, T)
            x = spec[..., np.newaxis]  # (n_mels, T, 1)
            x = np.expand_dims(x, axis=0)  # (1, n_mels, T, 1)

            probs = self.model.predict(x, verbose=0)[0]  # (num_classes,)
            probs = probs.astype(float)

            # convert to percentages
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
