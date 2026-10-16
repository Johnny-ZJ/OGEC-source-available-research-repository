# Public Release Scope

## Included

This repository includes the parts of the workflow needed to inspect and reproduce the evaluation logic:

1. expected data organization and schema checks;
2. subject-level cross-validation utilities;
3. fold-specific preprocessing utilities;
4. pooled out-of-fold metric calculation;
5. bootstrap confidence intervals;
6. paired nonparametric comparison utilities and Holm correction;
7. a stable interface documenting how the private OGEC implementation connects to the public evaluation pipeline.

## Not included

The following are intentionally withheld:

- full source code of the OGEC core neural architecture;
- full source code of the conditional WGAN-GP training module;
- non-public training heuristics and exact private implementation details;
- complete training hyperparameter files;
- pretrained model checkpoints.

## Rationale

The aim is to provide meaningful methodological transparency while protecting implementation details that remain part of ongoing research.

Users should not describe this repository as a complete end-to-end reproduction package.
