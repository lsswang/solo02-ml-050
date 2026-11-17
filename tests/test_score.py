"""Tests for global-optimal transcription.

The oracle here is fully independent of the dynamic-programming implementation:
it enumerates every legal note-event set per voice, pairs the voice schedules,
filters cross-voice constraints and evaluates the complete energy (including
all zero-onset frames) from scratch.
"""

import numpy as np
import pytest

import score


def all_notes(pitches, lo, hi, T, min_dur, max_dur):
    notes = []
    for p in pitches:
        if not (lo <= p <= hi):
            continue
        for s in range(T):
            for d in range(min_dur, max_dur + 1):
                if s + d <= T:
                    notes.append((p, s, s + d))
    return notes


def enumerate_schedules(notes, pitches, max_leap, T):
    """All non-overlapping schedules honoring the per-voice leap limit."""
    results = []

    def rec(idx, chosen, last_pitch):
        results.append(list(chosen))
        for k in range(idx, len(notes)):
            p, s, e = notes[k]
            if chosen and s < chosen[-1][2]:
                continue
            if last_pitch is not None and abs(p - last_pitch) > max_leap:
                continue
            rec(k + 1, chosen + [(p, s, e)], p)

    rec(0, [], None)
    return results


def schedule_matrices(sched_by_voice, pitches, T):
    P = len(pitches)
    idx = {p: i for i, p in enumerate(pitches)}
    B = np.zeros((P, T), dtype=np.int8)
    O = np.zeros((P, T), dtype=np.int8)
    motion = 0.0
    for v, sched in enumerate(sched_by_voice):
        last = None
        for p, s, e in sched:
            B[idx[p], s:e] = 1
            O[idx[p], s] = 1
            if last is not None:
                motion += abs(p - last)
            last = p
    return B, O, motion


def cross_legal(sched_by_voice, T):
    if len(sched_by_voice) == 1:
        return True
    v0, v1 = sched_by_voice
    for p0, s0, e0 in v0:
        for p1, s1, e1 in v1:
            lo, hi = max(s0, s1), min(e0, e1)
            if lo < hi:  # time overlap
                if p0 == p1 or p0 >= p1:
                    return False
    return True


def oracle(activations, onsets, request):
    T = activations.shape[1]
    voices = request["voices"]
    schedules = []
    for lo, hi in voices:
        notes = all_notes(request["pitches"], lo, hi, T,
                          request["min_duration"], request["max_duration"])
        schedules.append(enumerate_schedules(
            notes, request["pitches"], request["max_leap"], T))
    ow, nc, mw = request["onset_weight"], request["note_cost"], \
        request["motion_weight"]
    best = None
    best_energy = np.inf
    if len(schedules) == 1:
        combos = [(s, []) for s in schedules[0]]
    else:
        combos = ((s0, s1) for s0 in schedules[0] for s1 in schedules[1])
    count = 0
    for combo in combos:
        sched = [combo[0]] + ([combo[1]] if len(voices) == 2 else [])
        if not cross_legal(sched, T):
            continue
        B, O, motion = schedule_matrices(sched, request["pitches"], T)
        occ = float(np.sum((B.astype(float) - activations) ** 2))
        ons_e = float(np.sum((O.astype(float) - onsets) ** 2))
        n = sum(len(s) for s in sched)
        e = occ + ow * ons_e + nc * n + mw * motion
        count += 1
        if e < best_energy - 1e-12:
            best_energy = e
            best = (sched, B, O, n, motion, occ, ons_e)
    return best, best_energy, count


def base_request(pitches, voices, T, min_dur=1, max_dur=None, max_leap=12,
                 ow=1.0, nc=0.1, mw=0.0):
    return {"pitches": pitches, "voices": voices,
            "min_duration": min_dur, "max_duration": max_dur or T,
            "max_leap": max_leap, "onset_weight": ow, "note_cost": nc,
            "motion_weight": mw}


@pytest.mark.parametrize("seed", range(10))
def test_random_grids_match_oracle(seed):
    rng = np.random.default_rng(seed)
    P, T = 3, 5
    pitches = [60, 62, 64]
    V = 1 if seed % 2 else 2
    voices = [(60, 64)] if V == 1 else [(60, 62), (62, 64)]
    req = base_request(pitches, voices, T, min_dur=1, max_dur=4,
                       max_leap=8, ow=0.7, nc=0.2, mw=0.5)
    act = rng.random((P, T)) * 1.4
    ons = (rng.random((P, T)) > 0.7).astype(float)
    out = score.transcribe(act, ons, req)
    assert out["status"] == "optimal"
    (sched, B, O, n, motion, occ, ons_e), best_e, count = oracle(act, ons, req)
    assert count > 0
    assert abs(out["energy"] - best_e) <= 1e-8 + 1e-9 * abs(best_e)
    assert out["energies"]["occupancy"] == pytest.approx(occ, abs=1e-12)
    assert out["energies"]["onset"] == pytest.approx(ons_e, abs=1e-12)
    assert out["energies"]["note_count"] == n
    assert out["energies"]["motion"] == pytest.approx(motion, abs=1e-12)
    # Reported matrices and notes are mutually consistent.
    B2, O2, _ = schedule_matrices(
        [[(x["pitch"], x["start"], x["end"]) for x in out["notes"]
          if x["voice"] == v] for v in range(V)], pitches, T)
    np.testing.assert_array_equal(out["B"], B2)
    np.testing.assert_array_equal(out["O"], O2)


def test_same_pitch_reattack_is_two_notes():
    req = base_request([60], [(60, 60)], 4, min_dur=2, max_dur=4, ow=1.0)
    act = np.array([[1.0, 1.0, 1.0, 1.0]])
    ons = np.array([[1.0, 0.0, 1.0, 0.0]])
    out = score.transcribe(act, ons, req)
    notes = out["notes"]
    assert [n["start"] for n in notes] == [0, 2]
    assert notes[0]["end"] == notes[1]["start"]
    assert len(notes) == 2
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) < 1e-10


def test_rest_then_leap_and_leap_limit():
    # Leap of 7 is legal, leap of 12 rejected by max_leap=7.
    req = base_request([60, 67, 79], [(60, 79)], 6, min_dur=1, max_dur=2,
                       max_leap=7, ow=0.5, nc=0.0, mw=0.0)
    act = np.zeros((3, 6))
    act[0, 0:2] = 1.0
    act[1, 4:6] = 1.0
    ons = np.zeros((3, 6)); ons[0, 0] = 1.0; ons[1, 4] = 1.0
    out = score.transcribe(act, ons, req)
    used = sorted(n["pitch"] for n in out["notes"])
    assert used == [60, 67]
    # Rests do not clear the memory: a 19-semitone leap is forbidden.
    req2 = dict(req, max_leap=6)
    out2 = score.transcribe(act, ons, req2)
    assert all(n["pitch"] != 67 for n in out2["notes"]) or not any(
        n["pitch"] == 60 for n in out2["notes"])
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) < 1e-10
    _, best_e2, _ = oracle(act, ons, req2)
    assert abs(out2["energy"] - best_e2) < 1e-10


def test_two_voice_overlap_and_voice_ordering():
    req = base_request([60, 64], [(60, 64), (60, 64)], 2, min_dur=2,
                       max_dur=2, ow=1.0, nc=0.1)
    act = np.ones((2, 2))
    ons = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = score.transcribe(act, ons, req)
    assert len(out["notes"]) == 2
    by_voice = {n["voice"]: n["pitch"] for n in out["notes"]}
    assert by_voice[0] == 60 and by_voice[1] == 64
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) < 1e-10


def test_crossing_and_unison_are_forbidden():
    # Both voices can take 62; simultaneous unison is forbidden, and voice0
    # can never be above voice1.
    req = base_request([60, 62, 64], [(60, 64), (60, 64)], 3, min_dur=2,
                       max_dur=3, max_leap=12, ow=0.0, nc=0.0)
    act = np.zeros((3, 3)); act[1, :] = 1.0
    ons = np.zeros((3, 3))
    out = score.transcribe(act, ons, req)
    # At most one voice may hold pitch 62 at any frame.
    B = out["B"]
    assert np.all(B.sum(axis=0) <= 1)
    for t in range(3):
        active = [p for p in range(3) if B[p, t]]
        if len(active) == 2:
            assert active[0] < active[1]
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) < 1e-10


def test_range_exclusion():
    req = base_request([48, 72, 84], [(48, 60), (70, 84)], 4, min_dur=2,
                       max_dur=4, ow=0.5, nc=0.0)
    act = np.zeros((3, 4))
    act[0, :] = 1.0
    act[2, :] = 1.0
    ons = np.zeros((3, 4)); ons[0, 0] = 1.0; ons[2, 0] = 1.0
    out = score.transcribe(act, ons, req)
    pitches_by_voice = {v: {n["pitch"] for n in out["notes"]
                            if n["voice"] == v} for v in range(2)}
    assert pitches_by_voice[0] <= {48}
    assert pitches_by_voice[1] <= {72, 84}
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) < 1e-10


def test_empty_evidence_is_legal():
    req = base_request([60, 62], [(60, 62)], 4, min_dur=1, max_dur=4,
                       ow=1.0, nc=0.5)
    act = np.zeros((2, 4))
    ons = np.zeros((2, 4))
    out = score.transcribe(act, ons, req)
    assert out["notes"] == []
    assert out["status"] == "optimal"
    assert out["energy"] == 0.0
    np.testing.assert_array_equal(out["B"], np.zeros((2, 4), dtype=np.int8))
    np.testing.assert_array_equal(out["O"], np.zeros((2, 4), dtype=np.int8))


def test_onset_term_keeps_zero_frames():
    # Strong onset weight with a spurious onset at an unoccupied frame still
    # pays the constant (O-ons)^2 cost; check against the explicit oracle.
    req = base_request([60], [(60, 60)], 3, min_dur=1, max_dur=3,
                       ow=2.0, nc=1.5)
    act = np.array([[1.0, 1.0, 1.0]])
    ons = np.array([[0.0, 1.0, 0.0]])
    out = score.transcribe(act, ons, req)
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) <= 1e-8 + 1e-9 * abs(best_e)
    # The chosen onset position minimizes (O-ons)^2 over legal starts.
    assert out["notes"][0]["start"] == 1


def test_true_tie_both_optimal():
    # Symmetric evidence makes two placements exactly equal in energy.
    req = base_request([60, 62], [(60, 62)], 4, min_dur=2, max_dur=2,
                       ow=0.0, nc=0.0, mw=0.0)
    act = np.array([[1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0]]) * 0.5
    ons = np.zeros((2, 4))
    out = score.transcribe(act, ons, req)
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) < 1e-12
    # Deterministic, sorted output regardless of which tie is picked.
    keys = [(n["voice"], n["start"], n["end"], n["pitch"]) for n in out["notes"]]
    assert keys == sorted(keys)


def test_duration_constraints_exact():
    req = base_request([60], [(60, 60)], 5, min_dur=2, max_dur=3, ow=0.0)
    act = np.ones((1, 5))
    ons = np.zeros((1, 5)); ons[0, 0] = 1.0
    out = score.transcribe(act, ons, req)
    for n in out["notes"]:
        assert 2 <= n["end"] - n["start"] <= 3
        assert 0 <= n["start"] < n["end"] <= 5
    _, best_e, _ = oracle(act, ons, req)
    assert abs(out["energy"] - best_e) < 1e-10


def test_validation():
    req = base_request([60, 62], [(60, 62)], 3)
    good = np.ones((2, 3))
    with pytest.raises(ValueError):
        score.transcribe(np.full((2, 3), 4.0), good, req)  # external bound
    with pytest.raises(ValueError):
        score.transcribe(good, np.full((2, 3), 4.0), req)
    with pytest.raises(ValueError):
        score.transcribe(np.array([[np.nan, 0, 0], [0, 0, 0]]), good, req)
    bad = dict(req, pitches=[62, 60])
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)
    bad = dict(req, pitches=[60, 90])
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)
    bad = dict(req, voices=[(70, 60)])
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)
    bad = dict(req, min_duration=0)
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)
    bad = dict(req, max_duration=4)  # > T
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)
    bad = dict(req, max_leap=37)
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)
    bad = dict(req, onset_weight=2.1)
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)
    bad = dict(req, min_duration=1.5)
    with pytest.raises(ValueError):
        score.transcribe(good, good, bad)


def test_external_bound_does_not_apply_internally():
    req = base_request([60], [(60, 60)], 2, min_dur=1, max_dur=2)
    big = np.array([[3.4, 2.8]])  # legal internally, rejected externally
    with pytest.raises(ValueError):
        score.transcribe(big, np.zeros((1, 2)), req)
    out = score.transcribe(big, np.zeros((1, 2)), req, external=False)
    assert out["status"] == "optimal"


def test_input_not_mutated():
    req = base_request([60, 62], [(60, 62)], 3)
    act = np.ones((2, 3)); ons = np.zeros((2, 3))
    ac, oc = act.copy(), ons.copy()
    score.transcribe(act, ons, req)
    np.testing.assert_array_equal(act, ac)
    np.testing.assert_array_equal(ons, oc)
