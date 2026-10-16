from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import classification_metrics
from src.statistics import stratified_bootstrap_accuracy_ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    required = ["subject_id", "y_true", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prob_cols = [f"p{i}" for i in range(7)]
    y_prob = df[prob_cols].to_numpy() if all(c in df.columns for c in prob_cols) else None

    metrics = classification_metrics(
        df["y_true"].to_numpy(),
        df["y_pred"].to_numpy(),
        y_prob=y_prob,
    )

    lo, hi = stratified_bootstrap_accuracy_ci(
        df["y_true"].to_numpy(),
        df["y_pred"].to_numpy(),
        n_boot=args.bootstrap,
    )
    metrics["accuracy_bootstrap_ci95"] = [lo, hi]

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
