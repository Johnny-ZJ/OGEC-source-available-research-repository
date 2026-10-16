# OGEC — Source-Available Research Repository

This repository accompanies the manuscript:

**OGEC: Development and Internal Evaluation of a Unified Framework for Seven-Class Recognition from Resting-State QEEG**

## Purpose

This is a **partial, source-available research release** intended to improve transparency of the reported experiments while protecting unpublished implementation details of the OGEC core model.

The repository provides code for:

- BRMH/QEEG input-schema checking
- subject-level cross-validation split generation
- fold-specific preprocessing
- pooled out-of-fold metric calculation
- bootstrap confidence intervals
- paired statistical testing utilities
- interfaces for reproducing reported analyses from prediction files

## Scope of the public release

The following components are intentionally **not** included:

- the complete OGEC neural-network implementation
- the complete conditional WGAN-GP implementation
- private architecture internals beyond what is described in the manuscript
- full training-specific hyperparameterization
- trained model weights/checkpoints

The omitted components are part of an ongoing research implementation and are not required to inspect the evaluation protocol, leakage-control strategy, or statistical analysis code provided here.

This repository should therefore be cited as a **partial source-available research code release**, not as a complete end-to-end reproduction package.

## Data

The BRMH dataset is not redistributed here. Users should obtain the public dataset from its original source and comply with its applicable terms.

Expected QEEG organization in the manuscript:

- 6 frequency bands
- 19 PSD features per band
- 171 pairwise coherence features per band
- 1,140 QEEG features in total
- demographic covariates used in the main analysis: age, education, IQ

No participant-level data are included in this repository.

## Repository structure

```text
OGEC-source-available-research-repository/
├── README.md
├── RELEASE_SCOPE.md
├── CODE_AVAILABILITY.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── evaluation_example.json
├── src/
│   ├── __init__.py
│   ├── data_schema.py
│   ├── preprocessing.py
│   ├── splits.py
│   ├── metrics.py
│   ├── statistics.py
│   └── model_interface.py
├── scripts/
│   ├── check_dataset.py
│   └── evaluate_oof.py
└── results/
    └── README.md
```

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset-schema check

```bash
python scripts/check_dataset.py --csv /path/to/BRMH.csv
```

## Reproduce metrics from pooled OOF predictions

Prepare a CSV with the columns:

```text
subject_id,y_true,y_pred,p0,p1,p2,p3,p4,p5,p6
```

Then run:

```bash
python scripts/evaluate_oof.py --predictions /path/to/oof_predictions.csv
```

The script reports pooled accuracy, macro precision, macro recall, macro F1, and macro one-vs-rest AUC.

## Reproducibility note

The public code focuses on the parts of the study that can be independently audited without releasing the complete research implementation. The manuscript remains the authoritative description of the OGEC architecture.

## Contact

Please use the contact information provided in the manuscript for academic questions.
