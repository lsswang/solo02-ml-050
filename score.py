"""Global-optimal short-score transcription from frame-level evidence.

The transcriber never sees spectra or templates: it consumes arbitrary finite
nonnegative activation and onset matrices and searches the *complete* space of
legal note schedules.

A legal schedule obeys, simultaneously,

* every note has integer ``start < end`` and
  ``min_duration <= end-start <= max_duration``;
* notes of one voice never overlap (adjacent notes may touch, ``end == start``,
  which is a re-attack even at the same pitch);
* each pitch lies both in ``request['pitches']`` and in the voice range;
* one pitch is sounded by at most one voice at a time;
* when both voices sound, voice 0 is strictly below voice 1;
* consecutive notes of a voice (rests do not clear the memory) differ by at
  most ``max_leap`` semitones.

The minimized energy is

    E = sum (B - activations)^2
      + onset_weight * sum (O - onsets)^2
      + note_cost * #notes
      + motion_weight * sum |pitch_next - pitch_previous| ,

where B/O are binary occupancy/onset matrices built from the actual notes.
The onset term covers *every* frame, including frames without an onset, so
constant terms are never dropped.

Search: frame-synchronous dynamic programming over the joint voice state
(idle with remembered last pitch, or active on a pitch with a fixed end).
Every legal continuation is enumerated at every frame, so the result is a
global optimum; ties are broken deterministically and are still optimal.
"""

import numpy as np

_OCC = "occupancy"
_ONS = "onset"
_NOTE = "note_count"
_MOT = "motion"


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _matrix(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _weight(name, value):
    val = float(value)
    if not np.isfinite(val) or not (0.0 <= val <= 2.0):
        raise ValueError(f"{name} must be a finite value in [0, 2]")
    return val


def _validate(activations, onsets, request, external):
    act = _matrix("activations", activations)
    ons = _matrix("onsets", onsets)
    if act.shape != ons.shape:
        raise ValueError("activations and onsets must have the same shape")
    P, T = act.shape
    if P < 1 or T < 1:
        raise ValueError("activations must be non-empty")
    if external and np.any(act > 3.0):
        # The <= 3 bound only defines the independent-call test domain. The
        # combined pipeline calls with external=False and is never clipped.
        raise ValueError("external activations elements must be <= 3")
    if np.any(ons > 3.0):
        raise ValueError("onsets elements must be <= 3")

    if not isinstance(request, dict):
        raise ValueError("request must be a mapping")
    pitches = request.get("pitches")
    if not isinstance(pitches, (list, tuple)) or len(pitches) != P:
        raise ValueError("request['pitches'] must be a list of P ints")
    for p in pitches:
        if not _is_int(p) or not (48 <= p <= 84):
            raise ValueError("pitches must be integers in [48, 84]")
    if any(pitches[i] >= pitches[i + 1] for i in range(P - 1)):
        raise ValueError("pitches must be strictly increasing")

    voices = request.get("voices")
    if not isinstance(voices, (list, tuple)) or not (1 <= len(voices) <= 2):
        raise ValueError("request['voices'] must contain 1 or 2 ranges")
    voice_ranges = []
    for rng in voices:
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError("each voice range must be [low, high]")
        lo, hi = rng
        if not (_is_int(lo) and _is_int(hi) and 48 <= lo <= hi <= 84):
            raise ValueError("voice range endpoints must be ints in [48, 84]")
        voice_ranges.append((lo, hi))

    min_dur = request.get("min_duration")
    max_dur = request.get("max_duration")
    if not (_is_int(min_dur) and _is_int(max_dur)):
        raise ValueError("min_duration and max_duration must be integers")
    if not (1 <= min_dur <= max_dur <= T):
        raise ValueError("require 1 <= min_duration <= max_duration <= T")

    max_leap = request.get("max_leap")
    if not _is_int(max_leap) or not (0 <= max_leap <= 36):
        raise ValueError("max_leap must be an integer in [0, 36]")

    weights = {
        name: _weight(name, request.get(name, 0.0))
        for name in ("onset_weight", "note_cost", "motion_weight")
    }
    return act, ons, P, T, list(pitches), voice_ranges, min_dur, max_dur, \
        max_leap, weights


def transcribe(activations, onsets, request, external=True):
    """Find a globally optimal legal note schedule.

    Parameters
    ----------
    activations, onsets : array_like, shape (P, T)
        Arbitrary finite nonnegative evidence. Independent callers are bounded
        to entries <= 3; the combined spectral pipeline passes
        ``external=False`` because its own activations are not clipped.
    request : mapping
        ``pitches`` (P strictly increasing MIDI ints), one or two voice
        ranges, ``min_duration``, ``max_duration``, ``max_leap`` and the three
        weights in [0, 2].

    Returns
    -------
    dict with ``notes`` (sorted by voice/start/end/pitch), binary ``B`` and
    ``O``, per-term ``energy`` breakdown, total ``energy`` and ``status``.
    """
    act, ons, P, T, pitches, voice_ranges, min_dur, max_dur, max_leap, \
        weights = _validate(activations, onsets, request, external)
    V = len(voice_ranges)
    ow, nc, mw = weights["onset_weight"], weights["note_cost"], \
        weights["motion_weight"]

    allowed = [
        [p for p in range(P) if lo <= pitches[p] <= hi]
        for (lo, hi) in voice_ranges
    ]

    # Per-voice frame-independent start options: (pitch_idx, end, leap_ok is
    # checked later against the remembered previous pitch).
    start_options = []
    for v in range(V):
        opts = []
        for p in allowed[v]:
            for dur in range(min_dur, max_dur + 1):
                opts.append((p, dur))
        start_options.append(opts)

    # Voice state = (last_completed_pitch, active_note):
    #   last_completed_pitch is -1 before the first note and is remembered
    #   through rests (so rests never clear the leap history);
    #   active_note is None when idle, else (pitch_idx, end_frame).
    # The sounding pitch of an active note must not be confused with the
    # leap-history pitch, hence the two separate fields.  Each DP value stores
    # ``(cumulative_cost, note_tuple)`` so the notes realizing a state travel
    # with it: merging states can never corrupt the reconstruction.
    initial = tuple((-1, None) for _ in range(V))
    dp = {initial: (0.0, ())}

    for t in range(T):
        new_dp = {}
        for state, (cost, note_tuple) in dp.items():
            per_voice_choices = []
            for v in range(V):
                last, active = state[v]
                choices = [None]  # remain/turn idle (no start this frame)
                # A note ending exactly at frame t (half-open end == t) does
                # not occupy t, so a new note may start at t in the same
                # voice; this is the touching-note re-attack case.
                can_start = active is None or active[1] <= t
                if can_start:
                    for p, dur in start_options[v]:
                        end = t + dur
                        if end <= T and (
                                last == -1
                                or abs(pitches[p] - pitches[last]) <= max_leap):
                            choices.append((p, end))
                per_voice_choices.append(choices)

            def combine(v, acc):
                if v == V:
                    # Cross-voice legality on the pitches sounding at frame t.
                    sounding = []
                    for vv, ch in enumerate(acc):
                        _, active_v = state[vv]
                        if ch is not None:
                            sounding.append((vv, ch[0]))
                        elif active_v is not None and active_v[1] > t:
                            sounding.append((vv, active_v[0]))
                    idxs = [p for _, p in sounding]
                    if len(set(idxs)) != len(idxs):
                        return
                    if V == 2 and len(sounding) == 2:
                        if sounding[0][0] == 0:
                            p0, p1 = sounding[0][1], sounding[1][1]
                        else:
                            p0, p1 = sounding[1][1], sounding[0][1]
                        if pitches[p0] >= pitches[p1]:
                            return

                    # Frame-level evidence energy over every pitch.
                    occ = 0.0
                    onset = 0.0
                    note_add = 0.0
                    motion_add = 0.0
                    sounding_set = set(idxs)
                    starting = {}
                    for vv, ch in enumerate(acc):
                        if ch is not None:
                            starting[vv] = ch[0]
                    for p in range(P):
                        b = 1.0 if p in sounding_set else 0.0
                        d = b - act[p, t]
                        occ += d * d
                        o = 1.0 if p in starting.values() else 0.0
                        e = o - ons[p, t]
                        onset += e * e
                    for vv, p in starting.items():
                        note_add += nc
                        # The leap reference is the immediately previous note
                        # of this voice, including one ending at this same
                        # frame in a touching re-attack.
                        active_v = state[vv][1]
                        if active_v is not None and active_v[1] == t:
                            prev_pitch = active_v[0]
                        else:
                            prev_pitch = state[vv][0]
                        if prev_pitch != -1:
                            motion_add += mw * abs(
                                pitches[p] - pitches[prev_pitch])
                    frame_cost = occ + ow * onset + note_add + motion_add

                    nxt = []
                    new_notes = list(note_tuple)
                    for vv, ch in enumerate(acc):
                        last_v, active_v = state[vv]
                        if ch is not None:
                            # A new note starts. The "last note" used for the
                            # leap constraint is the immediately preceding note
                            # of this voice (including one ending this frame);
                            # remember the newly starting pitch right away so a
                            # later note measures its leap against this one.
                            nxt.append((ch[0], ch))
                            new_notes.append(
                                (vv, int(pitches[ch[0]]), int(t), int(ch[1])))
                        elif active_v is not None:
                            p, end = active_v
                            if end == t + 1:
                                nxt.append((p, None))  # completes after frame
                            else:
                                nxt.append((last_v, active_v))
                        else:
                            nxt.append((last_v, None))
                    nxt = tuple(nxt)
                    total = cost + frame_cost
                    incumbent = new_dp.get(nxt)
                    if incumbent is None or total < incumbent[0]:
                        new_dp[nxt] = (total, tuple(new_notes))
                    return
                for ch in per_voice_choices[v]:
                    acc.append(ch)
                    combine(v + 1, acc)
                    acc.pop()

            combine(0, [])
        dp = new_dp
        if not dp:
            return _unresolved(act, ons, P, T, weights)

    # At the end of the grid every active note must finish by frame T
    # (inclusive). A note starting on the final frame t=T-1 with end=T is
    # still represented as active in this state, because its completion
    # transition would occur at the nonexistent frame T; such states are
    # legal terminal states as long as the active note ends exactly at T.
    def terminal_ok(s):
        for _, active in s:
            if active is not None and active[1] != T:
                return False
        return True

    end_states = {s: val for s, val in dp.items() if terminal_ok(s)}
    if not end_states:
        return _unresolved(act, ons, P, T, weights)
    best_state = min(end_states, key=lambda s: end_states[s][0])
    dp_energy, note_tuple = end_states[best_state]

    notes = [{"voice": v, "pitch": p, "start": s, "end": e}
             for (v, p, s, e) in note_tuple]
    notes.sort(key=lambda n: (n["voice"], n["start"], n["end"], n["pitch"]))

    B, O, energy = _energy_from_notes(
        notes, act, ons, P, T, pitches, weights)
    if abs(energy["total"] - dp_energy) > 1e-9 + 1e-9 * max(
            1.0, abs(dp_energy)):
        # Internal inconsistency: never present an unverified optimum.
        return _unresolved(act, ons, P, T, weights)

    return {
        "notes": notes,
        "B": B,
        "O": O,
        "energy": float(energy["total"]),
        "energies": energy,
        "status": "optimal",
    }


def _energy_from_notes(notes, act, ons, P, T, pitches, weights):
    B = np.zeros((P, T), dtype=np.int8)
    O = np.zeros((P, T), dtype=np.int8)
    index = {midi: p for p, midi in enumerate(pitches)}
    last_pitch = {}
    motion = 0.0
    for n in sorted(notes, key=lambda x: (x["voice"], x["start"])):
        p = index[n["pitch"]]
        B[p, n["start"]:n["end"]] = 1
        O[p, n["start"]] = 1
        v = n["voice"]
        if v in last_pitch:
            motion += abs(n["pitch"] - last_pitch[v])
        last_pitch[v] = n["pitch"]
    occ = float(np.sum((B.astype(float) - act) ** 2))
    onset = float(np.sum((O.astype(float) - ons) ** 2))
    energy = {
        _OCC: occ,
        _ONS: onset,
        _NOTE: float(len(notes)),
        _MOT: float(motion),
    }
    total = (occ + weights["onset_weight"] * onset
             + weights["note_cost"] * len(notes)
             + weights["motion_weight"] * motion)
    return B, O, {**energy, "total": float(total)}


def _unresolved(act, ons, P, T, weights):
    empty = np.zeros((P, T), dtype=np.int8)
    occ = float(np.sum(act ** 2))
    onset = float(np.sum(ons ** 2))
    energy = {_OCC: occ, _ONS: onset, _NOTE: 0.0, _MOT: 0.0}
    total = occ + weights["onset_weight"] * onset
    return {
        "notes": [],
        "B": empty,
        "O": empty.copy(),
        "energy": float(total),
        "energies": {**energy, "total": float(total)},
        "status": "unresolved",
    }
