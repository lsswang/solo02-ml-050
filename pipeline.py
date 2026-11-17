"""Combined pipeline: joint spectral decomposition then global transcription.

The two stages stay decoupled exactly as specified:

* ``spectral.analyze`` only sees the short-time spectrum, calibrated templates
  and regularizers (no beat grid, no voice ranges);
* ``score.transcribe`` only sees activations, onsets and the score request
  (no spectrum, no templates, no re-fitting of a spectral model).

The activations produced here are passed to the *same* ``transcribe`` entry
point with ``external=False``: internally produced activations are not subject
to the independent-call ``<= 3`` test-domain bound and are never clipped.
"""

import numpy as np

import score
import spectral


def decompose_and_transcribe(templates, spectrum, onsets, request,
                             mask=None, reliability=None,
                             l1=0.0, ridge=0.5, temporal=0.0,
                             initial=None, max_iter=1000):
    """Run the spectral stage and transcribe its actual activations."""
    decomposition = spectral.analyze(
        templates, spectrum, mask=mask, reliability=reliability,
        l1=l1, ridge=ridge, temporal=temporal,
        initial=initial, max_iter=max_iter)
    transcription = score.transcribe(
        decomposition["activations"], onsets, request, external=False)
    return {
        "decomposition": decomposition,
        "transcription": transcription,
        # The spectral status is reported as-is: a non-converged stage is
        # never presented as a successful joint result.
        "spectral_converged": bool(decomposition["converged"]),
        "status": transcription["status"] if decomposition["converged"]
        else f"spectral_{decomposition['status']}",
    }


def schedule_energy(notes, activations, onsets, request):
    """Recompute the transcription energy of an arbitrary note schedule.

    Used for the frozen-table comparison: a note schedule found under old
    evidence is re-scored against the *current* activations with the current
    weights, instead of reusing a printed energy number.
    """
    act = np.asarray(activations, dtype=float)
    ons = np.asarray(onsets, dtype=float)
    P, T = act.shape
    pitches = list(request["pitches"])
    B = np.zeros((P, T), dtype=np.int8)
    O = np.zeros((P, T), dtype=np.int8)
    index = {midi: p for p, midi in enumerate(pitches)}
    last = {}
    motion = 0.0
    for n in sorted(notes, key=lambda x: (x["voice"], x["start"])):
        p = index[n["pitch"]]
        if B[p, n["start"]:n["end"]].any() or O[p, n["start"]]:
            raise ValueError("schedule is not a legal occupancy pattern")
        B[p, n["start"]:n["end"]] = 1
        O[p, n["start"]] = 1
        v = n["voice"]
        if v in last:
            motion += abs(n["pitch"] - last[v])
        last[v] = n["pitch"]
    occ = float(np.sum((B.astype(float) - act) ** 2))
    onset = float(np.sum((O.astype(float) - ons) ** 2))
    ow = float(request["onset_weight"])
    nc = float(request["note_cost"])
    mw = float(request["motion_weight"])
    total = occ + ow * onset + nc * len(notes) + mw * motion
    return {
        "energy": total,
        "energies": {
            "occupancy": occ,
            "onset": onset,
            "note_count": float(len(notes)),
            "motion": float(motion),
            "total": total,
        },
        "B": B,
        "O": O,
    }
