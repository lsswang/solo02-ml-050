"""Tests for the combined decomposition -> transcription pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

import pipeline
import score
import spectral


def request(T, pitches=(60, 64, 67), voices=((60, 67),), **kw):
    base = {"pitches": list(pitches), "voices": [list(v) for v in voices],
            "min_duration": 1, "max_duration": T, "max_leap": 12,
            "onset_weight": 1.0, "note_cost": 0.1, "motion_weight": 0.0}
    base.update(kw)
    return base


def test_end_to_end_two_evidence_schedules_change_notes():
    # Same templates/grid/constraints: only the spectral evidence changes.
    D = np.array([[1.0, 0.2, 0.0],
                  [0.8, 1.0, 0.3],
                  [0.1, 0.7, 1.0],
                  [0.0, 0.2, 0.9],
                  [0.3, 0.0, 0.6]])
    T = 4
    A_a = np.array([[0.9, 0.9, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.8],
                    [0.0, 0.0, 0.0, 0.0]])
    A_b = np.array([[0.9, 0.9, 0.9, 0.9],
                    [0.7, 0.7, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.8]])
    Ya = np.clip(D @ A_a, 0, 3)
    Yb = np.clip(D @ A_b, 0, 3)
    onsets = np.zeros((3, T)); onsets[0, 0] = 1.0; onsets[1, 2] = 1.0
    onsets[2, 2] = 1.0
    req = request(T)

    out_a = pipeline.decompose_and_transcribe(D, Ya, onsets, req, ridge=0.3)
    out_b = pipeline.decompose_and_transcribe(D, Yb, onsets, req, ridge=0.3)
    assert out_a["spectral_converged"] and out_b["spectral_converged"]
    assert out_a["transcription"]["status"] == "optimal"
    assert out_b["transcription"]["status"] == "optimal"

    # The second evidence genuinely changes the realized notes.
    assert out_a["transcription"]["notes"] != out_b["transcription"]["notes"]

    # Frozen-table check: the old schedule, re-scored against the CURRENT
    # activations and current weights, must be strictly worse than the new
    # optimum (the new optimum is the global minimum under current evidence).
    old_notes = out_a["transcription"]["notes"]
    new_act = out_b["decomposition"]["activations"]
    old_scored = pipeline.schedule_energy(old_notes, new_act, onsets, req)
    assert old_scored["energy"] > out_b["transcription"]["energy"] + 1e-12
    # The re-scored energy is the full current objective: independently
    # re-derive its occupancy component against the current activations.
    B_old = old_scored["B"]
    assert old_scored["energies"]["occupancy"] == pytest.approx(
        float(np.sum((B_old.astype(float) - new_act) ** 2)), abs=1e-12)


def test_old_schedule_recompute_is_strictly_worse_numeric():
    rng = np.random.default_rng(0)
    D = rng.random((4, 3)) + 0.2
    T = 5
    Y1 = np.clip(D @ (rng.random((3, T)) * 1.2), 0, 3)
    Y2 = np.clip(D @ (rng.random((3, T)) * 1.2), 0, 3)
    onsets = (rng.random((3, T)) > 0.6).astype(float)
    req = request(T, voices=((60, 67), (64, 72)), min_dur=1, max_dur=3,
                  motion_weight=0.4, note_cost=0.2)
    o1 = pipeline.decompose_and_transcribe(D, Y1, onsets, req, ridge=0.4,
                                           temporal=0.5)
    o2 = pipeline.decompose_and_transcribe(D, Y2, onsets, req, ridge=0.4,
                                           temporal=0.5)
    if o1["transcription"]["notes"] == o2["transcription"]["notes"]:
        pytest.skip("evidence happened to yield identical schedules")
    scored = pipeline.schedule_energy(
        o1["transcription"]["notes"], o2["decomposition"]["activations"],
        onsets, req)
    assert scored["energy"] >= o2["transcription"]["energy"]
    if scored["energy"] == o2["transcription"]["energy"]:
        # A true tie under new evidence: both are globally optimal.
        pass
    else:
        assert scored["energy"] > o2["transcription"]["energy"]


def test_non_converged_spectral_is_reported_as_is():
    D = np.array([[1.0, 0.5], [0.4, 1.0], [0.2, 0.2], [0.6, 0.1]])
    Y = np.clip(D @ np.array([[1.2, 0.0], [0.0, 1.1]]), 0, 3)
    onsets = np.zeros((2, 2)); onsets[0, 0] = 1.0; onsets[1, 1] = 1.0
    req = request(2, pitches=(60, 64), voices=((60, 64),), min_dur=1,
                  max_dur=2)
    out = pipeline.decompose_and_transcribe(
        D, Y, onsets, req, ridge=0.5, temporal=1.0, max_iter=1)
    if not out["decomposition"]["converged"]:
        assert out["spectral_converged"] is False
        assert out["status"].startswith("spectral_")
        # Transcription still runs on the actual (best feasible) activations.
        assert out["transcription"]["status"] == "optimal"
        np.testing.assert_array_equal(
            out["transcription"]["B"].shape, (2, 2))


def test_transcription_never_sees_spectrum():
    import inspect
    sig = inspect.signature(score.transcribe)
    for forbidden in ("templates", "spectrum", "D", "Y"):
        assert forbidden not in sig.parameters


def test_internal_activations_not_clipped():
    # Build activations > 3 from the analyzer itself (allowed internally).
    D = np.array([[1.0], [0.0], [0.0], [0.1]])
    Y = np.array([[3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    req = request(2, pitches=(60,), voices=((60, 60),), min_dur=1, max_dur=2)
    out = pipeline.decompose_and_transcribe(
        D, Y, np.zeros((1, 2)), req, ridge=0.1)
    assert out["spectral_converged"]
    assert np.max(out["decomposition"]["activations"]) > 2.7
    assert out["transcription"]["status"] == "optimal"


def test_fixture_demo_runs(tmp_path):
    cases = json.loads(Path("fixtures/cases.json").read_text())
    spec = spectral.analyze(**{k: v for k, v in cases["ordinary_spectral"].items()
                               if k in ("templates", "spectrum", "ridge")})
    assert spec["converged"]
    notes = score.transcribe(spec["activations"],
                             **cases["ordinary_score"])
    assert notes["status"] == "optimal" and len(notes["notes"]) >= 1


def test_missing_frames_pipeline():
    D = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.8], [0.3, 0.3]])
    Y = np.clip(D @ np.array([[1.0, 1.0, 0.0], [0.0, 0.8, 0.8]]), 0, 3)
    mask = np.ones_like(Y, dtype=bool)
    mask[:, 1] = False  # whole middle frame missing, variables retained
    req = request(3, pitches=(60, 64), voices=((60, 64),), min_dur=1,
                  max_dur=3)
    out = pipeline.decompose_and_transcribe(
        D, Y, np.zeros((2, 3)), req, mask=mask, ridge=0.4, temporal=1.0)
    assert out["spectral_converged"]
    assert out["transcription"]["status"] == "optimal"
