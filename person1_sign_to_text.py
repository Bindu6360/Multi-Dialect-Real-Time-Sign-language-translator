"""Fallback sign-language module for the SignBridge demo.

This project ships with a placeholder implementation so the dashboard can start
without the full ASL model pipeline. The real sign-recognition logic can be
added later without changing the rest of the application.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

SHARED_OUTPUT_FILE = "shared_output.txt"


class DummyClassifier:
    """Minimal classifier compatible with the dashboard's expectations."""

    def predict(self, features):
        count = 1 if np.asarray(features).ndim == 1 else len(features)
        return np.array(["A"] * count, dtype=object)

    def predict_proba(self, features):
        arr = np.asarray(features)
        rows = 1 if arr.ndim == 1 else len(arr)
        return np.ones((rows, 1), dtype=float)


def load_or_build_classifier():
    return DummyClassifier(), ["A"]


def extract_features(landmarks):
    # Return a fixed-size placeholder vector matching the dashboard's usage.
    return np.zeros(21 * 3, dtype=float)


class WordBuilder:
    def __init__(self):
        self._buffer = ""

    def update(self, token):
        token = str(token).strip()
        if not token:
            return
        self._buffer += token
        if len(self._buffer) > 40:
            self._buffer = self._buffer[-40:]

    def get_display(self):
        return self._buffer

    def reset(self):
        self._buffer = ""


def write_shared_output(text):
    Path(SHARED_OUTPUT_FILE).write_text(str(text), encoding="utf-8")
