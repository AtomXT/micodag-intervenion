# Running batch - not a completed comparison

The sequential 194-fit local batch was launched on 2026-09-06 UTC. The first
worker log began at 01:40:47 UTC. At this implementation handoff, job 0
(`scm_4`, PS-MIP complete, graph multiplier 1/16) was still optimizing; no
operating point had finished. It had a feasible incumbent but was not yet
certified to the frozen 1e-4 relative-gap tolerance.

The initial process identities were parent PID 87816 and worker PID 87827.
These are historical identifiers, not proof that the processes remain alive.
Verify the current command lines before any process-specific action. The
execution session identifier at launch was 52202.

The original four-hour PS-MIP and one-hour baseline time limits are unchanged.
The batch may take days. The machine must remain running for local progress.
No Quest jobs were submitted. Automatic follow-up checks were offered to the
user but have not been authorized or created at this handoff.

## Completed implementation checks

- Both configurations prepared: 40,000 observations each, all ten variables,
  23-edge reference, no filtering or rank transformation.
- Numeric-only archive acquisition read 36,255,821 bytes rather than the
  4.6 GB image archive. Source licenses, revisions, CRC32 and SHA256 are saved.
- Exhaustive numerical preflight completed for all 40,960 local optima per
  configuration. The frozen bounds are recorded in the data and run design.
- All 23 chamber tests passed. Combined regression: 133/134 passed, with the
  known unrelated BaCaDI Quest wall-time mismatch recorded in verification.json.
- All 15 preview PDF pages were rendered; plot layout variants and diagnostics
  were visually checked. The empty preview is explicitly marked incomplete.
  Actual recovery points still require visual inspection after fitting.
- Frozen fitting code and data identity were validated after the run started.
  Only new chamber files and outputs were added; existing Sachs and core
  optimizer files were not edited for this pilot.

## Remaining work

1. Let the approved sequential run account for all 194 fits. Check for failures
   or a stopped runner without launching duplicate fits. Resume the same driver
   only if it is no longer running and identities still match; existing failures
   are preserved, not silently retried.
2. Run the final audit and plotting commands in experiments/CAUSAL_CHAMBERS_PILOT.md.
   Final export requires all fits accounted for, including explicit failures.
3. Inspect the actual full/sparse/rate and PS-MIP-only figures and verify the
   reference, physical/programmed splits, target assignments, solver diagnostics,
   and ambiguity counts. Explain comparative results without assuming a winner.
4. Deliver the completed plot bundle and assessment. Do not present the existing
   summary_preview folder as a completed comparison.

The pre-fit diagnostics already indicate approximate linearity but appreciable
residual dependence and changing non-target residual variances. The largest
absolute residual correlations are about 0.728 (scm_4) and 0.566 (scm_5).
Non-target residual variance ratios span about 0.243-1.178 and 0.626-1.172,
respectively. These are model limitations, not grounds for filtering the data.
