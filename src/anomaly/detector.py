# src/anomaly/detector.py

import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False

    def fit(self, X: list):
        """
        Fit the model with normal usage data.
        X should be a list of [energy, token_count, flops_per_layer]
        """
        self.model.fit(X)
        self.is_fitted = True

    def detect(self, sample: list) -> (bool, str):
        """
        Detect if a sample is an anomaly.
        Returns: (is_anomaly, reason)
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector not fitted yet.")

        pred = self.model.predict([sample])[0]  # -1 for anomaly, 1 for normal
        is_anomaly = pred == -1

        # Reasoning (just basic thresholds for explainability)
        _, token_count, flops_per_layer = sample
        reasons = []
        if token_count > 100:
            reasons.append("Excessive token count")
        if flops_per_layer > 1e18:
            reasons.append("High FLOPs per layer")

        reason = ", ".join(reasons) if is_anomaly else "Within normal bounds"
        return is_anomaly, reason