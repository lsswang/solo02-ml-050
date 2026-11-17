# Joint nonnegative spectral decomposition and global short-score transcription

Python 3.11 CPU project. Inputs are synthetic nonnegative short-time spectra
with calibrated templates and an integer time grid. There is no waveform
analysis, audio dataset, listening test or pretrained/downloaded model.

## Layout

| File | Role |
| --- | --- |
| `spectral.py` | `analyze(templates, spectrum, mask, reliability, l1, ridge, temporal, initial, max_iter)` — joint nonnegative decomposition of every frame against all templates at once. |
| `score.py` | `transcribe(activations, onsets, request, external=True)` — global-optimal legal note schedule from frame evidence; no spectrum or template input. |
| `pipeline.py` | `decompose_and_transcribe(...)` wiring the two stages, plus `schedule_energy(...)` for frozen-table re-scoring. |
| `demo.py` | Reproducible scenarios; writes `fixtures/products.json`. |
| `tests/` | Independent oracles and property tests (156 tests). |

## Reproducible commands

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python demo.py
.venv/bin/python -m pytest tests/ -q
```

## Spectral stage (`spectral.analyze`)

The single global objective over `A[P,T] >= 0` is

```
J(A) = .5 sum_{f,t} mask[f,t] W[f,t] (sum_p D[f,p] A[p,t] - Y[f,t])^2
     + l1 sum A + ridge/2 sum A^2
     + temporal/2 sum_{p,t>0} (A[p,t] - A[p,t-1])^2 .
```

`mask=False` is a missing bin (not an observed zero): its data weight becomes
zero, while the activation variables and all regularizers for that frame stay
in the problem. Because `ridge > 0`, the objective is strictly convex and the
minimizer is unique.

### Computation and intermediate state

* Per frame the weighted data term is assembled as a small quadratic form
  `0.5 a_t^T H_t a_t - g_t^T a_t` with `H_t = D^T diag(mask_t W_t) D + ridge I`
  (matrices are at most 4x4). Temporal regularization adds a block-tridiagonal
  coupling between adjacent frames.
* The only optimization state is the full activation matrix `A` (plus an
  extrapolation point). Frames are never solved independently and smoothed,
  and no single pitch is fixed beforehand.
* Solver: restarted accelerated projected gradient (FISTA) with Armijo
  backtracking, followed by an exact active-set Newton polish on the **same**
  objective. FISTA identifies the zero/nonzero support (the slow part under the
  nonsmooth l1 term); the polish solves the strictly convex smooth KKT equations
  on the free coordinates and re-admits/drops coordinates from full KKT checks.
* Convergence is decided solely by the projected-gradient residual
  `max |A - max(A - grad J(A), 0)| <= 1e-7`, evaluated on the full nonnegative
  constraint. Optimizer-internal success flags are never used. A capped budget
  returns `iteration_limit`, a non-finite line search returns
  `numerical_failure`; in both cases the best feasible point seen since
  initialization is returned and its objective is never more than `1e-9` above
  the initial objective.

Returns the actual `activations`, per-term `costs`, total `objective`, the full
same-shape `gradient`, the KKT `residual`, `status`, `converged`, iteration
counts and the initial objective. Activations are solutions to the stated
spectral objective, not calibrated note probabilities.

## Transcription stage (`score.transcribe`)

The transcriber consumes arbitrary finite nonnegative `activations` and
`onsets` of shape `[P,T]` and never sees `Y` or `D`. Independent callers are
limited to entries `<= 3`; the combined pipeline calls with `external=False`
because internally produced activations are not clipped.

A legal schedule enforces, simultaneously: integer `0 <= start < end <= T`,
`min_duration <= end-start <= max_duration`, no overlap within a voice
(touching `end == start` is allowed and is a re-attack even at equal pitch),
pitch inside both the voice range and `request['pitches']`, one pitch in at
most one voice at a time, strict `voice0 < voice1` when both sound, and
`max_leap` between consecutive notes of a voice where rests do not clear the
previous pitch.

The globally minimized energy is

```
E = sum (B - activations)^2
  + onset_weight sum (O - onsets)^2
  + note_cost * #notes
  + motion_weight * sum_voice |pitch_next - pitch_previous| ,
```

with binary occupancy `B` and onset `O` built from the actual notes. The onset
sum covers every frame, including frames with no onset, so its constant terms
are never dropped.

### Computation and intermediate state

* Frame-synchronous dynamic programming over the joint voice state. Each voice
  state stores `(last_pitch, active_note)` separately: the last note (remembered
  across rests for the leap constraint) must not be confused with the currently
  sounding note. An active note is `(pitch, end)` and is dropped on completion.
* At every frame every legal per-voice choice is enumerated — stay idle, let an
  active note continue, or start a note of any allowed pitch/duration (including
  a same-frame touching re-attack). Voice choices are combined and filtered by
  the cross-voice unison/ordering rules before adding that frame's occupancy,
  onset, note and motion costs.
* Each DP state carries the full note tuple realizing it, so state merging can
  never corrupt reconstruction. With `P <= 4`, `T <= 8` and at most two voices
  the complete search is small; stored state scales with the number of reachable
  joint states times their note lists, bounded by the tiny grid. The result is a
  global optimum; ties are broken deterministically and remain optimal. If an
  internal consistency check ever fails, the function returns
  `status='unresolved'` rather than labeling a local answer optimal.

Returns sorted `notes`, `B`, `O`, per-term energies, total `energy` and
`status`.

## Combined pipeline and frozen-table comparison

`pipeline.decompose_and_transcribe` passes the analyzer's actual activations to
the same `transcribe` entry point (`external=False`). A non-converged spectral
stage is reported as `spectral_iteration_limit` / `spectral_numerical_failure`
rather than claimed successful; transcription still runs on the best feasible
activations. `pipeline.schedule_energy` re-scores an old note schedule against
the current activations and current weights, so a frozen old table is shown to
be strictly worse under new evidence instead of reusing a stale printed energy.

## Verification

* Spectral objectives/gradients are recomputed by a standalone implementation
  and checked against central finite differences; minimizers are verified both
  by exhaustive active-set enumeration of the small QP (an independent exact
  route) and by SciPy L-BFGS-B, with explicit projected-gradient KKT checks.
* Transcription is checked against a fully independent exhaustive enumeration
  of every legal note-event set and voice assignment, evaluating the complete
  energy from scratch.
* Coverage includes ordinary single-note spectra and note merges, harmonic
  overlap, correlated templates, partially and fully missing frames, temporal
  coupling and boundary KKT, same-pitch re-attacks, leaps after rests, two-voice
  overlap, crossing/unison prohibition, range exclusion, empty scores, true
  ties and external evidence. No claim is made about real-recording accuracy.
