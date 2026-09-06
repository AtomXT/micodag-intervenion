# Chamber graphical-lasso screening audit

Observational/reference rows only; all 10 variables and 10,000 reference rows retained. The exact synthetic rule is c=1 in alpha=c*sqrt(log(p)/n_ref). The constant grid and two numerical settings were fixed before fitting. Reference graphs enter evaluation only.

The strict setting changes numerical accuracy, not the graphical-lasso objective. Warnings and unsuccessful estimates are preserved, not accepted as verified screens. The historical rule is primary; this audit does not choose a penalty using reference recovery.

| Configuration | Solver | c | Candidate pairs / 45 | Parent sets / 5120 | Reference retained / 23 | Physical / 21 | Programmed / 2 | Numerically clean |
|---|---|---:|---:|---:|---:|---:|---:|---|
| scm_4 | existing_defaults | 0.125 | failed | — | — | — | — | no |
| scm_4 | existing_defaults | 0.25 | 36 | 1984 | 21 | 19 | 2 | False |
| scm_4 | existing_defaults | 0.5 | 37 | 2304 | 21 | 19 | 2 | False |
| scm_4 | existing_defaults | 1 | 32 | 1472 | 20 | 18 | 2 | False |
| scm_4 | existing_defaults | 2 | 32 | 1312 | 17 | 16 | 1 | True |
| scm_4 | existing_defaults | 4 | 35 | 1680 | 19 | 18 | 1 | True |
| scm_4 | strict_diagnostic | 0.125 | 39 | 2560 | 21 | 19 | 2 | False |
| scm_4 | strict_diagnostic | 0.25 | 36 | 1984 | 21 | 19 | 2 | True |
| scm_4 | strict_diagnostic | 0.5 | 37 | 2304 | 21 | 19 | 2 | True |
| scm_4 | strict_diagnostic | 1 | 32 | 1472 | 20 | 18 | 2 | True |
| scm_4 | strict_diagnostic | 2 | 32 | 1312 | 17 | 16 | 1 | True |
| scm_4 | strict_diagnostic | 4 | 35 | 1680 | 19 | 18 | 1 | True |
| scm_5 | existing_defaults | 0.125 | failed | — | — | — | — | no |
| scm_5 | existing_defaults | 0.25 | 39 | 2752 | 22 | 20 | 2 | False |
| scm_5 | existing_defaults | 0.5 | 35 | 1888 | 19 | 18 | 1 | False |
| scm_5 | existing_defaults | 1 | 33 | 1312 | 17 | 17 | 0 | False |
| scm_5 | existing_defaults | 2 | 30 | 800 | 15 | 15 | 0 | True |
| scm_5 | existing_defaults | 4 | 29 | 656 | 14 | 14 | 0 | True |
| scm_5 | strict_diagnostic | 0.125 | failed | — | — | — | — | no |
| scm_5 | strict_diagnostic | 0.25 | 39 | 2752 | 22 | 20 | 2 | False |
| scm_5 | strict_diagnostic | 0.5 | 35 | 1888 | 19 | 18 | 1 | False |
| scm_5 | strict_diagnostic | 1 | 33 | 1312 | 17 | 17 | 0 | True |
| scm_5 | strict_diagnostic | 2 | 30 | 800 | 15 | 15 | 0 | True |
| scm_5 | strict_diagnostic | 4 | 29 | 656 | 14 | 14 | 0 | True |

## Edges excluded by the historical c=1 rule

- scm_4, existing_defaults: red -> vis_1; blue -> ir_2; blue -> ir_3.
- scm_4, strict_diagnostic: red -> vis_1; blue -> ir_2; blue -> ir_3.
- scm_5, existing_defaults: red -> green; red -> vis_1; blue -> green; blue -> ir_1; blue -> ir_2; blue -> ir_3.
- scm_5, strict_diagnostic: red -> green; red -> vis_1; blue -> green; blue -> ir_1; blue -> ir_2; blue -> ir_3.

## Interpretation and limits

A documented edge absent from the screen cannot be recovered by the subsequent MIP, regardless of its orientation. Complete reference retention is only an empirical check: the physical reference does not certify every unreported effect absent. A screened optimization certificate would apply only to the restricted candidate space. Fewer parent sets are not a measured wall-time speedup; MIP search time can vary substantially.

All screens, precision estimates, excluded edges, warnings, timing, versions and identities are saved in results.json and the individual rows. The original unscreened pilot, input arrays, core optimizer, synthetic defaults and running jobs are unchanged.

Reproduce with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-dcdi/bin/python -B experiments/audit_chamber_glasso.py
```
