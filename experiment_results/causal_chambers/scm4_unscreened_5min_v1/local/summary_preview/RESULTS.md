# Causal Chambers pilot results

Accounted for 0/97 planned fits; 0 successful. PS-MIP: 0/0 accounted fits certified optimal within the frozen 1e-4 MIP-gap tolerance.

## Protocol

Each included configuration retains all 40,000 observations and the same ten variables. One reference-derived affine transformation is shared across environments; there is no rank transformation, row filtering, or subsampling. IGSP and UT-IGSP use Gaussian invariance tests, unlike the Sachs kernel-test protocol. These are controlled physical/hybrid experiments, not biological validation. Each reference combines 21 documented physical sensor links with two programmed actuator links.

The CSVs and generator contain 10,000 observations per environment; the README's inherited 1,000-observation description is not used.

## Sparse comparison (descriptive, not penalty selection)

Entries are directed TP/FP, followed by skeleton TP/FP. An unattained budget is reported as such. Ties use directed FP, skeleton TP, skeleton FP, then the smaller tuning value. Full paths remain in the bundle.

### scm_4

| Method | FP <= 3 | FP <= 5 | Minimum directed FP | Successful/planned |
|---|---|---|---:|---:|
| PS-MIP: complete targets | Unattained | Unattained | None | 0/17 |
| PS-MIP: known-present targets | Unattained | Unattained | None | 0/17 |
| PS-MIP: hidden targets (diagnostic) | Unattained | Unattained | None | 0/17 |
| UT-IGSP: targets supplied | Unattained | Unattained | None | 0/8 |
| GnIES: target union supplied | Unattained | Unattained | None | 0/15 |
| GIES: complete targets | Unattained | Unattained | None | 0/15 |
| IGSP: complete targets | Unattained | Unattained | None | 0/8 |

Physical versus programmed recovery at the descriptive FP <= 5 points:

| Method | Physical directed TP / 21 | Programmed directed TP / 2 | Directed TP with ambiguous orientation |
|---|---:|---:|---:|

## Model suitability

The intervention protocol is clearer than Sachs: incoming actuator links are explicitly removed at intervention targets. However, real sensor noise is not the same as the programmed actuator noise. The following pre-fit diagnostics were recorded before recovery evaluation:

| Configuration | Largest absolute residual correlation | Non-target residual variance ratio range | Largest held-out quadratic improvement |
|---|---:|---:|---:|
| scm_4 | 0.728 | 0.243-1.178 | 0.042% |

Small quadratic improvements support approximate linearity of the measured relationships on this protocol; they do not establish an exactly linear-Gaussian model. Residual correlation conflicts with independent errors, and changes in non-target residual variance conflict with shared non-target noise. Correlated sensor noise, unrepresented effects, and changing noise levels are plausible explanations, not uniquely identified causes.

Some interventions on actuator roots principally change means. PS-MIP's centered score does not exploit pure mean shifts; the hidden-target diagnostic can therefore miss documented root interventions even when the graph is recovered well. This is different from pretending the experimenter lacks target labels: the complete and known-present cases retain them.

GnIES permits changes in target noise without requiring all incoming coefficients to be removed, whereas PS-MIP uses hard interventions and shared non-target variances. Performance differences must be interpreted together with these model and information differences, not as a controlled comparison of optimization alone.

## Numerical bounds and completion

- scm_4: 40,960 local optima checked without graph truth; maximum required diagonal 8.939, maximum coefficient magnitude 8.716. Frozen bounds: {'gamma_lower': 1e-05, 'gamma_upper': 89.39484844716586, 'coefficient_bound': 87.16218669853644}.

The original upper bounds of 10 already cover these preflight extrema; the tenfold-margin bounds do not explain any recovery gain. The optimizer and synthetic defaults were not changed. Successful PS-MIP artifacts include the fitted precision factor, incumbent objective, objective bound, solver status, gap, and inactive-bound checks.

## Points outside the 0-25 FP viewing window

The display limit is applied independently per panel; full-range figures and raw artifacts preserve all settings.

| Configuration | Method | Directed outside | Skeleton outside |
|---|---|---:|---:|
| scm_4 | PS-MIP: complete targets | 0 | 0 |
| scm_4 | PS-MIP: known-present targets | 0 | 0 |
| scm_4 | PS-MIP: hidden targets (diagnostic) | 0 | 0 |
| scm_4 | UT-IGSP: targets supplied | 0 | 0 |
| scm_4 | GnIES: target union supplied | 0 | 0 |
| scm_4 | GIES: complete targets | 0 | 0 |
| scm_4 | IGSP: complete targets | 0 | 0 |

## Failures and nonoptimal solutions

No failed fits among the accounted results.
Nonoptimal feasible PS-MIP fits: 0.

## Runtime and solver diagnostics

All fits start fresh. Times below include accounted failures; missing fits are not counted as zero-runtime successes. PS-MIP optimality is certification within the numerical tolerance, not a guarantee of correct causal recovery.

| Configuration | Method | Successful / planned | Certified PS-MIP | Nonoptimal | Failed / timed out | Worker minutes | Median / maximum PS-MIP gap (%) |
|---|---|---:|---:|---:|---:|---:|---|
| scm_4 | PS-MIP: complete targets | 0/17 | 0 | 0 | 0 | 0.000 | not applicable / unavailable |
| scm_4 | PS-MIP: known-present targets | 0/17 | 0 | 0 | 0 | 0.000 | not applicable / unavailable |
| scm_4 | PS-MIP: hidden targets (diagnostic) | 0/17 | 0 | 0 | 0 | 0.000 | not applicable / unavailable |
| scm_4 | UT-IGSP: targets supplied | 0/8 | not applicable | 0 | 0 | 0.000 | not applicable / unavailable |
| scm_4 | GnIES: target union supplied | 0/15 | not applicable | 0 | 0 | 0.000 | not applicable / unavailable |
| scm_4 | GIES: complete targets | 0/15 | not applicable | 0 | 0 | 0.000 | not applicable / unavailable |
| scm_4 | IGSP: complete targets | 0/8 | not applicable | 0 | 0 | 0.000 | not applicable / unavailable |

Frozen timing policy: {'fit_limit_seconds': 300, 'worker_watchdog_seconds': 360, 'ps_mip_limit_scope': 'native solver; precomputation and result saving reported separately', 'baseline_limit_scope': 'entire method adapter with an interrupt timer', 'grace_purpose': 'startup, solver return and result saving; no planned additional search'}. Raw rows report algorithm/solver overruns, and attempts.jsonl reports external watchdog use and total worker time.


## Interpretation and limitations

This pilot assesses model suitability and recovery, not whether one method can be made to win. Directed counts use saved DAG representatives and can depend on unresolved orientations; graph-class ambiguity counts are retained. Missing edges in the physical reference are not proven biological or physical non-effects. No AUC, artificial envelope, bootstrap confirmation, target validation, or false-discovery guarantee is claimed. Raw TP counts are not compared directly with Sachs's 18-edge reference.

## Sources

- [Dataset and protocols](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_camera_v1)
- [Official camera reference adjacency](https://github.com/juangamella/causal-chamber-package/tree/main/causalchamber/ground_truth/adjacencies)
- [Causal Chambers paper](https://www.nature.com/articles/s42256-024-00964-x)
