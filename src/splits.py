from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_subject_level_folds(
    subject_ids,
    labels,
    n_splits: int = 10,
    random_state: int = 2026,
):
    """Return subject-level stratified outer folds.

    Each subject must appear exactly once in the input arrays.
    """
    subject_ids = np.asarray(subject_ids)
    labels = np.asarray(labels)

    if len(subject_ids) != len(labels):
        raise ValueError("subject_ids and labels must have identical length.")
    if len(np.unique(subject_ids)) != len(subject_ids):
        raise ValueError("subject_ids must be unique at the subject-table level.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(subject_ids, labels), start=1
    ):
        folds.append(
            {
                "fold": fold_id,
                "train_subjects": subject_ids[train_idx].tolist(),
                "test_subjects": subject_ids[test_idx].tolist(),
            }
        )
    return folds
