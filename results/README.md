# Results files

Participant-level prediction files are not bundled in this repository.

To reproduce pooled evaluation statistics, place a prediction file locally with
the following columns:

```text
subject_id,y_true,y_pred,p0,p1,p2,p3,p4,p5,p6
```

For privacy and release-control reasons, prediction files are ignored by
`.gitignore` by default. If the authors later choose to release a de-identified
OOF prediction table, that decision should be made separately and only after
confirming that such release is consistent with the source dataset terms.
