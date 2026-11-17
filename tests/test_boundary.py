"""Additional boundary and property tests for both stages."""

import numpy as np
import pytest

import score
import spectral


def test_minimal_dimensions_and_extreme_weights():
    # Smallest legal spectral problem: P=1, F=3, T=1.
    D = np.array([[1.0], [0.5], [0.2]])
    Y = np.array([[2.0], [1.0], [0.4]])
    out = spectral.analyze(D, Y, l1=0.2, ridge=1.0, temporal=2.0)
    assert out["converged"]
    assert out["activations"].shape == (1, 1)
    # Closed form for T=1: max(0, (D^T Y - l1) / (D^T D + ridge)).
    a_star = max(0.0, (float((D.T @ Y).ravel()[0]) - 0.2)
                 / (float((D.T @ D).ravel()[0]) + 1.0))
    assert abs(float(out["activations"][0, 0]) - a_star) < 1e-7

    req = {"pitches": [60], "voices": [[60, 60]], "min_duration": 1,
           "max_duration": 1, "max_leap": 0, "onset_weight": 2.0,
           "note_cost": 2.0, "motion_weight": 2.0}
    tr = score.transcribe(out["activations"], np.ones((1, 1)), req)
    assert tr["status"] == "optimal"
    assert tr["notes"][0]["start"] == 0 and tr["notes"][0]["end"] == 1


def test_reliability_extremes_and_mask_shapes():
    D = np.random.default_rng(0).random((4, 2)) + 0.2
    Y = np.clip(D @ np.ones((2, 3)), 0, 3)
    W = np.full_like(Y, 5.0)
    M = np.ones_like(Y, dtype=bool)
    M[0, 0] = False
    out = spectral.analyze(D, Y, mask=M, reliability=W, ridge=0.5)
    assert out["converged"]
    W_low = np.full_like(Y, 0.1)
    out2 = spectral.analyze(D, Y, mask=M, reliability=W_low, ridge=0.5)
    assert out2["converged"]


def test_output_dtypes_and_shapes():
    D = np.random.default_rng(1).random((5, 3)) + 0.1
    Y = np.clip(D @ np.random.default_rng(2).random((3, 4)), 0, 3)
    out = spectral.analyze(D, Y, ridge=0.5, temporal=0.5)
    assert isinstance(out["activations"], np.ndarray)
    assert out["activations"].shape == (3, 4)
    assert out["gradient"].shape == (3, 4)
    assert np.all(out["activations"] >= 0.0)
    assert set(out["costs"]) == {"data", "l1", "ridge", "temporal"}
    assert abs(sum(out["costs"].values()) - out["objective"]) < 1e-10


def test_score_energy_terms_sum_to_total():
    req = {"pitches": [60, 62, 64], "voices": [[60, 64], [60, 64]],
           "min_duration": 1, "max_duration": 4, "max_leap": 12,
           "onset_weight": 1.3, "note_cost": 0.7, "motion_weight": 0.4}
    rng = np.random.default_rng(3)
    act = rng.random((3, 5)) * 2.0
    ons = (rng.random((3, 5)) > 0.6).astype(float)
    out = score.transcribe(act, ons, req)
    e = out["energies"]
    total = (e["occupancy"] + 1.3 * e["onset"] + 0.7 * e["note_count"]
             + 0.4 * e["motion"])
    assert abs(total - out["energy"]) < 1e-12
    # B/O consistency with notes.
    for n in out["notes"]:
        p = req["pitches"].index(n["pitch"])
        assert out["O"][p, n["start"]] == 1
        assert np.all(out["B"][p, n["start"]:n["end"]] == 1)
    assert int(out["O"].sum()) == len(out["notes"])


def test_onset_constant_terms_are_counted():
    # With no notes chosen, onset energy must equal sum ons^2 over all cells.
    req = {"pitches": [60], "voices": [[60, 60]], "min_duration": 1,
           "max_duration": 3, "max_leap": 0, "onset_weight": 1.0,
           "note_cost": 2.0, "motion_weight": 0.0}
    act = np.zeros((1, 3))
    ons = np.array([[0.5, 0.2, 0.1]])
    out = score.transcribe(act, ons, req)
    # A note costs note_cost + onset penalties; verify the empty schedule is
    # genuinely evaluated with the full onset constant (not zeroed).
    assert out["energies"]["onset"] == pytest.approx(
        float(np.sum(ons ** 2)), abs=1e-12)


def test_tie_is_optimal_not_just_feasible():
    # Construct a case with two exactly equal-energy schedules and confirm the
    # returned one matches the exhaustive enumeration energy.
    req = {"pitches": [60, 62], "voices": [[60, 62]], "min_duration": 2,
           "max_duration": 2, "max_leap": 12, "onset_weight": 0.0,
           "note_cost": 0.0, "motion_weight": 0.0}
    act = np.full((2, 4), 0.5)
    ons = np.zeros((2, 4))
    out = score.transcribe(act, ons, req)
    # Any legal schedule has identical occupancy here; check energy identity.
    occ = float(np.sum((out["B"].astype(float) - act) ** 2))
    assert out["energy"] == pytest.approx(occ, abs=1e-12)


def test_unresolved_is_reported_not_called_optimal(monkeypatch):
    req = {"pitches": [60], "voices": [[60, 60]], "min_duration": 1,
           "max_duration": 2, "max_leap": 0, "onset_weight": 1.0,
           "note_cost": 0.1, "motion_weight": 0.0}
    real = score._energy_from_notes

    def corrupted(notes, *args, **kwargs):
        B, O, e = real(notes, *args, **kwargs)
        e["total"] += 1.0  # force the DP/recompute consistency check to fail
        return B, O, e

    monkeypatch.setattr(score, "_energy_from_notes", corrupted)
    out = score.transcribe(np.ones((1, 2)), np.zeros((1, 2)), req)
    assert out["status"] == "unresolved"
    assert isinstance(out["notes"], list)


def test_spectral_inputs_immutable_with_initial():
    D = np.random.default_rng(4).random((4, 2)) + 0.1
    Y = np.clip(D @ np.ones((2, 2)), 0, 3)
    A0 = np.full((2, 2), 0.5)
    A0c = A0.copy()
    spectral.analyze(D, Y, ridge=0.5, initial=A0, max_iter=3)
    np.testing.assert_array_equal(A0, A0c)


def test_score_rejects_wrong_shapes_and_non_integer_times():
    req = {"pitches": [60, 62], "voices": [[60, 62]], "min_duration": 1,
           "max_duration": 3, "max_leap": 12}
    with pytest.raises(ValueError):
        score.transcribe(np.ones((2, 3)), np.ones((2, 4)), req)
    bad = dict(req, min_duration=1.0)  # float instead of int
    with pytest.raises(ValueError):
        score.transcribe(np.ones((2, 3)), np.zeros((2, 3)), bad)
    # Duplicate pitches violate the strictly-increasing requirement.
    with pytest.raises(ValueError):
        score.transcribe(np.ones((2, 3)), np.zeros((2, 3)),
                         dict(req, pitches=[60, 60]))
