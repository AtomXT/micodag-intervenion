# Screening decision

Do not replace the two-configuration pilot with either tested graphical-lasso
screen: neither retains every documented reference edge in both configurations.
The original unscreened pilot, data, optimizer and synthetic defaults remain
unchanged. No running experiment was stopped or reconfigured.

## Primary check: exact synthetic screening rule

Use only the 10,000 reference observations and all ten variables, with
`alpha = sqrt(log(10)/10000) = 0.015174271293851463` and support threshold `1e-8`.
Both orientations of each retained pair would be allowed. The same production
helper used in the synthetic experiments was checked directly. Default numerical
settings emitted convergence warnings; a separate tighter numerical solve
converged cleanly at this penalty in both configurations and returned the same
supports.

| Configuration | Candidate pairs / 45 | Parent sets / 5120 | Reference edges retained / 23 |
|---|---:|---:|---:|
| scm_4 | 32 | 1472 | 20 |
| scm_5 | 33 | 1312 | 17 |

The excluded reference edges are:

- scm_4: `red -> vis_1`, `blue -> ir_2`, `blue -> ir_3`.
- scm_5: `red -> green`, `red -> vis_1`, `blue -> green`, `blue -> ir_1`,
  `blue -> ir_2`, `blue -> ir_3`. Both programmed links are excluded.

A pre-frozen sensitivity grid also checked multipliers 1/8, 1/4, 1/2, 1, 2,
and 4, with the default and stricter numerical settings: 24 fits total. None of
the returned observational screens retained all 23 edges. Three fits failed;
all failures and warnings remain in the detailed results. This finite grid does
not prove that every possible graphical-lasso penalty would fail.

## Conservative follow-up: union across environments

After the primary audit failed, a separate fixed-c=1 follow-up fit a screen in
each of the four environments and took their union. The rule uses measurements
and environment memberships, not the reference graph or intervention target
labels. All eight strict component fits converged cleanly. Sixteen component fits
were accounted for including the default-numerics comparison.

| Configuration | Candidate pairs / 45 | Parent sets / 5120 | Reference edges retained / 23 |
|---|---:|---:|---:|
| scm_4 | 42 | 3712 | 23 |
| scm_5 | 38 | 2624 | 20 |

The union passes the empirical coverage check for scm_4 only, removing three
candidate pairs and 27.5% of parent sets. It still excludes `blue -> ir_1`,
`blue -> ir_2`, and `blue -> ir_3` in scm_5. Parent-set reduction is not a measured
runtime speedup.

## Interpretation

Graphical lasso screens conditional association, not physical causation directly.
Overlapping sensor signals and the previously reported residual dependence make
the loss of documented edges a practical concern. As an illustration, in scm_5
the reference-sample blue/ir_3 marginal correlation is 0.499035, but its
unpenalized partial correlation conditional on the other eight variables is
0.001267. This is a descriptive illustration, not proof of why every edge was
excluded or of a valid independent-noise causal model.

An edge removed from the candidate superstructure cannot subsequently be
recovered. We did not add excluded edges back from the reference graph or choose
penalties to optimize reference recovery. Complete retention of the documented
graph would be an empirical check, not a guarantee about undocumented effects;
screened solver certificates would apply only to the restricted search space.

## Verification and reproduction

All returned screen counts and all four environment unions were recomputed from
saved adjacencies. Both original prepared-data identities and the running pilot's
code identity match their frozen design. All 28 screening/chamber tests pass.

Details: [observational audit](RESULTS.md) and
[environment-union audit](../glasso_union_audit_v1/RESULTS.md). Each directory
contains frozen settings, individual outputs, precision estimates, warnings,
failures, runtimes and package/code identities.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv-dcdi/bin/python -B experiments/audit_chamber_glasso.py
.venv-dcdi/bin/python -B experiments/audit_chamber_glasso_union.py
.venv-dcdi/bin/python -B -m unittest tests.test_chamber_glasso_audit tests.test_chamber_pilot -v
```
