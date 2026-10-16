from __future__ import annotations

from abc import ABC, abstractmethod


class OGECModelInterface(ABC):
    """Public interface to the non-released OGEC core implementation.

    The concrete model implementation is intentionally not included in the
    public repository.
    """

    @abstractmethod
    def fit(self, X_qeeg, X_demo, y):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X_qeeg, X_demo):
        raise NotImplementedError

    def predict(self, X_qeeg, X_demo):
        prob = self.predict_proba(X_qeeg, X_demo)
        return prob.argmax(axis=1)
