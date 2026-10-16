from __future__ import annotations

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class QEEGFoldPreprocessor:
    """Fold-specific QEEG clipping and standardization.

    Fit only on the training partition of a fold, then apply unchanged to
    validation/test partitions.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = float(iqr_multiplier)
        self.lower_ = None
        self.upper_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: np.ndarray) -> "QEEGFoldPreprocessor":
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.iqr_multiplier * iqr
        self.upper_ = q3 + self.iqr_multiplier * iqr
        Xc = np.clip(X, self.lower_, self.upper_)
        self.scaler_.fit(Xc)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        X = np.asarray(X, dtype=float)
        Xc = np.clip(X, self.lower_, self.upper_)
        return self.scaler_.transform(Xc)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class DemographicFoldPreprocessor:
    """Fold-specific standardization followed by KNN imputation.

    Both fitted components must be estimated using training participants only.
    """

    def __init__(self, n_neighbors: int = 5):
        self.scaler_ = StandardScaler()
        self.imputer_ = KNNImputer(
            n_neighbors=n_neighbors,
            weights="uniform",
            metric="nan_euclidean",
        )

    def fit(self, X: np.ndarray) -> "DemographicFoldPreprocessor":
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.fit_transform(X)
        self.imputer_.fit(Xs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X)
        return self.imputer_.transform(Xs)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
