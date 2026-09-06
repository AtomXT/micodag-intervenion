# Environment-union graphical-lasso follow-up

Proposed after observational-only screening excluded documented edges. Use c=1 for every environment: alpha=sqrt(log(p)/n_e). Fit each screen using its own standardized measurements, then take the undirected union. The reference graph and target labels do not enter fitting or the union operation. This is a post-hoc diagnostic, not a change to the frozen experiment.

| Configuration | Solver | Candidate pairs / 45 | Parent sets / 5120 | Reference retained / 23 | Numerically clean |
|---|---|---:|---:|---:|---|
| scm_4 | existing_defaults | 42 | 3712 | 23 | False |
| scm_4 | strict_diagnostic | 42 | 3712 | 23 | True |
| scm_5 | existing_defaults | 38 | 2624 | 20 | False |
| scm_5 | strict_diagnostic | 38 | 2624 | 20 | True |

## Remaining exclusions

- scm_4, existing_defaults: none.
- scm_4, strict_diagnostic: none.
- scm_5, existing_defaults: blue -> ir_1; blue -> ir_2; blue -> ir_3.
- scm_5, strict_diagnostic: blue -> ir_1; blue -> ir_2; blue -> ir_3.

Complete reference retention would be empirical, not a guarantee about undocumented effects. A union is more conservative than any component but can still miss an adjacency everywhere. A group with any failed or nonconverged component is not treated as a verified screen. Reduced parent-set counts do not establish solver runtime improvement.

The original pilot, all fitting inputs, core optimizer, synthetic defaults and running jobs are unchanged.

Reproduce: `.venv-dcdi/bin/python -B experiments/audit_chamber_glasso_union.py` with the four numerical-library thread variables set to 1.
