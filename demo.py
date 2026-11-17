"""Reproducible demonstrations of the joint spectral -> score pipeline.

Run with the project virtualenv:

    python demo.py

Everything is synthetic nonnegative short-time spectra; no audio, waveform
analysis or pretrained model is involved.  Structured results are written to
``fixtures/products.json`` and a short summary is printed.
"""

import json
from pathlib import Path

import numpy as np

import pipeline
import score
import spectral

ROOT = Path(__file__).parent
CASES = json.loads((ROOT / "fixtures" / "cases.json").read_text())


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def summarize_spectral(out):
    return {
        "status": out["status"],
        "converged": out["converged"],
        "iterations": out["iterations"],
        "objective": out["objective"],
        "residual": out["residual"],
        "costs": out["costs"],
        "activations": to_jsonable(np.round(out["activations"], 6)),
    }


def main():
    products = {}

    # 1. Ordinary single-note spectral explanation, then a legal note merge.
    ordinary = spectral.analyze(**CASES["ordinary_spectral"])
    ordinary_notes = score.transcribe(
        ordinary["activations"], **CASES["ordinary_score"])
    products["ordinary"] = {
        "spectral": summarize_spectral(ordinary),
        "score": to_jsonable(ordinary_notes),
    }

    # 2. Harmonic overlap: several pitches explain one frame jointly.
    overlap_cfg = CASES["overlap"]
    D = np.array(overlap_cfg["templates"])
    A_true = np.array([[0.9, 0.0], [0.7, 1.0], [0.0, 0.8]])
    Y = np.clip(D @ A_true, 0.0, 3.0)
    overlap = spectral.analyze(D, Y, ridge=overlap_cfg["ridge"], temporal=0.3)
    products["harmonic_overlap"] = {
        "spectral": summarize_spectral(overlap),
        "simultaneous_pitches_per_frame":
            [int(np.sum(overlap["activations"][:, t] > 0.05))
             for t in range(Y.shape[1])],
    }

    # 3. Correlated templates with a fully missing middle frame: variables and
    #    regularizers at that frame are retained, only data evidence is absent.
    rng = np.random.default_rng(0)
    Dc = np.array([[1.0, 0.95], [0.9, 1.0], [0.3, 0.4], [0.1, 0.2],
                   [0.4, 0.05]])
    Ac = rng.random((2, 4)) * 1.2
    Yc = np.clip(Dc @ Ac, 0.0, 3.0)
    mask = np.ones_like(Yc, dtype=bool)
    mask[:, 2] = False
    missing = spectral.analyze(Dc, Yc, mask=mask, ridge=0.4, temporal=0.8)
    products["missing_frame"] = {
        "spectral": summarize_spectral(missing),
        "missing_frame_activations_finite":
            bool(np.all(np.isfinite(missing["activations"][:, 2]))),
    }

    # 4. Temporal coupling changes the activation without changing the target.
    coupling = CASES["coupling"]
    Dt = np.array(coupling["templates"])
    Yb = np.array(coupling["spectrum_before"])
    flat = spectral.analyze(Dt, Yb, ridge=coupling["ridge"], temporal=0.0)
    smooth = spectral.analyze(Dt, Yb, ridge=coupling["ridge"], temporal=2.0)
    products["temporal_coupling"] = {
        "uncoupled": summarize_spectral(flat),
        "coupled": summarize_spectral(smooth),
        "middle_frame_activation": {
            "uncoupled": float(flat["activations"][0, 1]),
            "coupled": float(smooth["activations"][0, 1]),
        },
    }

    # 5. Same-pitch re-attack: threshold merging would collapse two notes.
    repeat = score.transcribe(**CASES["repeat"])
    products["same_pitch_reattack"] = to_jsonable(repeat)

    # 6. Two-voice overlap with strict voice ordering.
    polyphony = score.transcribe(**CASES["polyphony"])
    products["two_voice_overlap"] = to_jsonable(polyphony)

    # 7. Combined pipeline with two evidence schedules and the frozen-table
    #    check: the old note schedule is re-scored under current evidence.
    T = 4
    Dp = np.array([[1.0, 0.2, 0.0], [0.8, 1.0, 0.3], [0.1, 0.7, 1.0],
                   [0.0, 0.2, 0.9], [0.3, 0.0, 0.6]])
    A_a = np.array([[0.9, 0.9, 0.0, 0.0], [0.0, 0.0, 0.8, 0.8],
                    [0.0, 0.0, 0.0, 0.0]])
    A_b = np.array([[0.9, 0.9, 0.9, 0.9], [0.7, 0.7, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.8]])
    Ya = np.clip(Dp @ A_a, 0.0, 3.0)
    Yb = np.clip(Dp @ A_b, 0.0, 3.0)
    onsets = np.zeros((3, T))
    onsets[0, 0] = onsets[1, 2] = onsets[2, 2] = 1.0
    request = {"pitches": [60, 64, 67], "voices": [[60, 67]],
               "min_duration": 1, "max_duration": T, "max_leap": 12,
               "onset_weight": 1.0, "note_cost": 0.1, "motion_weight": 0.0}
    run_a = pipeline.decompose_and_transcribe(Dp, Ya, onsets, request,
                                              ridge=0.3)
    run_b = pipeline.decompose_and_transcribe(Dp, Yb, onsets, request,
                                              ridge=0.3)
    old_scored = pipeline.schedule_energy(
        run_a["transcription"]["notes"],
        run_b["decomposition"]["activations"], onsets, request)
    products["combined_evidence_change"] = {
        "run_a": {
            "spectral_status": run_a["decomposition"]["status"],
            "notes": to_jsonable(run_a["transcription"]["notes"]),
            "energy": run_a["transcription"]["energy"],
        },
        "run_b": {
            "spectral_status": run_b["decomposition"]["status"],
            "notes": to_jsonable(run_b["transcription"]["notes"]),
            "energy": run_b["transcription"]["energy"],
        },
        "frozen_old_schedule_under_current_evidence": {
            "energy": old_scored["energy"],
            "strictly_worse_than_current_optimum":
                bool(old_scored["energy"]
                     > run_b["transcription"]["energy"] + 1e-12),
        },
    }

    # 8. A deliberately capped spectral run reports its honest status.
    capped = spectral.analyze(Dp, Yb, ridge=0.5, temporal=1.0, max_iter=1)
    products["capped_spectral_run"] = {
        "status": capped["status"],
        "converged": capped["converged"],
        "objective_not_worse_than_initial":
            bool(capped["objective"] <= capped["initial_objective"] + 1e-9),
    }

    out_path = ROOT / "fixtures" / "products.json"
    out_path.write_text(json.dumps(to_jsonable(products), indent=2))

    for name, item in products.items():
        print(f"== {name} ==")
        print(json.dumps(item, indent=2)[:1200])
        print()
    print(f"Structured products written to {out_path}")


if __name__ == "__main__":
    main()
