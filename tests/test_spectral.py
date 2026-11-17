"""Tests for the joint nonnegative spectral decomposition.

Independent verification routes:

* objective/gradient values are recomputed from a standalone implementation
  and the gradient is additionally checked with central finite differences;
* minimizers are cross-checked against a separate SciPy L-BFGS-B solve of the
  same objective plus explicit complementarity/KKT checks, never against the
  projected-gradient implementation's own status.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

import spectral


def raw_objective(A, D, Y, M, W, l1, ridge, temporal):
    A = A.reshape(D.shape[1], Y.shape[1])
    R = D @ A - Y
    P, T = A.shape
    data = 0.5 * np.sum(M * W * R * R)
    l1c = l1 * np.sum(A)
    rc = 0.5 * ridge * np.sum(A * A)
    tc = 0.0
    if T > 1:
        d = A[:, 1:] - A[:, :-1]
        tc = 0.5 * temporal * np.sum(d * d)
    return data + l1c + rc + tc, dict(data=data, l1=l1c, ridge=rc, temporal=tc)


def raw_gradient(A, D, Y, M, W, l1, ridge, temporal):
    A = A.reshape(D.shape[1], Y.shape[1])
    P, T = A.shape
    R = D @ A - Y
    g = D.T @ (M * W * R) + l1 + ridge * A
    if T > 1:
        lap = np.empty_like(A)
        lap[:, 0] = A[:, 0] - A[:, 1]
        lap[:, -1] = A[:, -1] - A[:, -2]
        if T > 2:
            lap[:, 1:-1] = 2 * A[:, 1:-1] - A[:, :-2] - A[:, 2:]
        g += temporal * lap
    return g


def make_case(seed=0, F=5, P=3, T=4, missing=False, all_missing=False):
    rng = np.random.default_rng(seed)
    D = rng.random((F, P)) * 2.5 + 0.05
    A_true = (rng.random((P, T)) * 1.5) * (rng.random((P, T)) > 0.4)
    Y = D @ A_true + rng.random((F, T)) * 0.05
    Y = np.clip(Y, 0, 3)
    M = np.ones((F, T), dtype=bool)
    if all_missing:
        M[:] = False
    elif missing:
        M[rng.random((F, T)) < 0.3] = False
    W = np.clip(rng.random((F, T)) * 2 + 0.2, 0.1, 5.0)
    return D, Y, M, W


def lbfsb_minimize(D, Y, M, W, l1, ridge, temporal, A0):
    P, T = A0.shape
    fun = lambda a: raw_objective(a, D, Y, M, W, l1, ridge, temporal)[0]
    jac = lambda a: raw_gradient(a, D, Y, M, W, l1, ridge, temporal).ravel()
    res = minimize(fun, A0.ravel(), jac=jac, method="L-BFGS-B",
                   bounds=[(0.0, None)] * (P * T),
                   options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 5000})
    return res.x.reshape(P, T), res.fun


def kkt_residual(A, D, Y, M, W, l1, ridge, temporal):
    g = raw_gradient(A, D, Y, M, W, l1, ridge, temporal)
    return float(np.max(np.abs(A - np.maximum(A - g, 0.0))))


def exact_qp_oracle(D, Y, M, W, l1, ridge, temporal):
    """Guaranteed small-QP oracle: enumerate active sets (n <= 32) and solve
    the unconstrained KKT equations for each, checking full complementarity.
    The smooth Hessian is positive definite (ridge > 0), so any KKT point is
    the unique global minimizer.
    """
    P, T = D.shape[1], Y.shape[1]
    n = P * T
    K = np.zeros((n, n))
    q = np.zeros(n)
    Mt = M.astype(float) * W
    for t in range(T):
        w = Mt[:, t]
        Dw = D * w[:, None]
        Ht = D.T @ Dw + ridge * np.eye(P)
        # Smooth objective at frame t: .5 x^T Ht x + q_t^T x with the linear
        # part q = -(D W)^T Y + l1, so stationarity on free coords is Kx+q=0.
        qt = -Dw.T @ Y[:, t] + l1
        sl = slice(t * P, (t + 1) * P)
        K[sl, sl] = Ht
        q[sl] = qt
        if temporal > 0.0 and t > 0:
            prev = slice((t - 1) * P, t * P)
            K[sl, prev] -= temporal * np.eye(P)
            K[prev, sl] -= temporal * np.eye(P)
            K[sl, sl] += temporal * np.eye(P)
            K[prev, prev] += temporal * np.eye(P)

    best = None
    for mask_free in range(1 << n):
        free = np.array([bool(mask_free & (1 << i)) for i in range(n)])
        x = np.zeros(n)
        if free.any():
            try:
                xf = np.linalg.solve(K[np.ix_(free, free)], -q[free])
            except np.linalg.LinAlgError:
                continue
            if np.any(xf < -1e-10):
                continue
            x[free] = np.maximum(xf, 0.0)
        # Frame-major (Fortran) ordering matches the K block assembly where
        # each frame's P pitches occupy a contiguous slice.
        A = x.reshape(P, T, order="F")
        if kkt_residual(A, D, Y, Mt, np.ones_like(W), l1, ridge,
                        temporal) < 1e-9:
            f = raw_objective(A, D, Y, Mt, np.ones_like(W), l1, ridge,
                              temporal)[0]
            if best is None or f < best[0]:
                best = (f, A)
    assert best is not None
    return best[1], best[0]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("temporal", [0.0, 0.7, 2.0])
@pytest.mark.parametrize("l1", [0.0, 0.05, 0.2])
def test_random_cases_against_lbfgsb(seed, temporal, l1):
    D, Y, M, W = make_case(seed=seed, missing=(seed % 2 == 0))
    ridge = 0.3
    out = spectral.analyze(D, Y, mask=M, reliability=W,
                           l1=l1, ridge=ridge, temporal=temporal)
    assert out["converged"], out["status"]
    # Guaranteed oracle: exhaustive active-set enumeration of the small QP.
    A_exact, f_exact = exact_qp_oracle(D, Y, M, W, l1, ridge, temporal)
    assert np.max(np.abs(out["activations"] - A_exact)) < 1e-7
    assert abs(out["objective"] - f_exact) <= 1e-8 + 1e-8 * abs(f_exact)
    # Independent cross-check via a different algorithm (L-BFGS-B).
    A_ref, f_ref = lbfsb_minimize(D, Y, M, W, l1, ridge, temporal,
                                  np.zeros_like(out["activations"]))
    assert np.max(np.abs(out["activations"] - A_ref)) < 2e-6
    assert kkt_residual(out["activations"], D, Y, M, W, l1, ridge,
                        temporal) <= 1e-9


def test_objective_terms_and_gradient_values():
    D, Y, M, W = make_case(seed=3)
    l1, ridge, temporal = 0.07, 0.4, 0.9
    out = spectral.analyze(D, Y, mask=M, reliability=W,
                           l1=l1, ridge=ridge, temporal=temporal)
    A = out["activations"]
    f, terms = raw_objective(A, D, Y, M, W, l1, ridge, temporal)
    g = raw_gradient(A, D, Y, M, W, l1, ridge, temporal)
    assert abs(out["objective"] - f) <= 1e-8 + 1e-8 * abs(f)
    for key in terms:
        assert abs(out["costs"][key] - terms[key]) <= 1e-8 + 1e-8 * abs(f)
    np.testing.assert_allclose(out["gradient"], g, atol=1e-8, rtol=1e-8)


def test_gradient_finite_differences():
    D, Y, M, W = make_case(seed=7)
    l1, ridge, temporal = 0.03, 0.5, 1.1
    P, T = D.shape[1], Y.shape[1]
    rng = np.random.default_rng(11)
    A0 = rng.random((P, T)) * 2.0
    # Build the same quadratic form the analyzer uses and evaluate the
    # gradient at the fixed probe point (no optimizer iteration involved).
    Mw = M.astype(float) * W
    H = np.empty((P, P, T))
    Gt = np.empty((P, T))
    for t in range(T):
        Dw = D * Mw[:, t][:, None]
        H[:, :, t] = D.T @ Dw + ridge * np.eye(P)
        Gt[:, t] = Dw.T @ Y[:, t]
    _, g, _ = spectral._objective_and_gradient(
        A0, H, Gt, D, Mw, Y, l1, ridge, temporal)
    h = 1e-6
    for p in range(P):
        for t in range(T):
            Ap = A0.copy(); Ap[p, t] += h
            Am = A0.copy(); Am[p, t] -= h
            fp = raw_objective(Ap, D, Y, M, W, l1, ridge, temporal)[0]
            fm = raw_objective(Am, D, Y, M, W, l1, ridge, temporal)[0]
            assert abs((fp - fm) / (2 * h) - g[p, t]) < 1e-6


def test_all_missing_frames_retain_variables():
    # With no data evidence the problem separates over cells; temporal couples
    # frames. All-missing still yields a finite, unique, KKT-valid solution.
    D, Y, _, W = make_case(seed=2, all_missing=True)
    M = np.zeros(Y.shape, dtype=bool)
    for temporal in (0.0, 1.0):
        out = spectral.analyze(D, Y, mask=M, reliability=W, l1=0.05,
                               ridge=0.5, temporal=temporal)
        assert out["converged"]
        assert np.all(np.isfinite(out["activations"]))
        # l1 with no data -> zero activation is optimal at ridge > 0.
        assert np.max(np.abs(out["activations"])) < 1e-8
        assert kkt_residual(out["activations"], D, Y, M, W, 0.05, 0.5,
                            temporal) <= 1e-7


def test_correlated_templates_and_harmonic_overlap():
    # Nearly collinear templates plus two simultaneously active pitches.
    D = np.array([[1.0, 0.9, 0.0],
                  [1.0, 1.0, 0.2],
                  [0.2, 0.3, 1.0],
                  [0.0, 0.1, 1.0]])
    A_true = np.array([[0.8, 0.0], [0.6, 1.0], [0.0, 0.7]])
    Y = np.clip(D @ A_true, 0, 3)
    M = np.ones_like(Y, dtype=bool)
    W = np.ones_like(Y)
    out = spectral.analyze(D, Y, mask=M, reliability=W, l1=0.0,
                           ridge=0.2, temporal=0.3)
    assert out["converged"]
    A_ref, f_ref = lbfsb_minimize(D, Y, M, W, 0.0, 0.2, 0.3,
                                  np.zeros_like(A_true))
    assert np.max(np.abs(out["activations"] - A_ref)) < 1e-7
    # Joint decomposition recovers simultaneous activations (not one template).
    assert np.count_nonzero(out["activations"] > 0.05) >= 3


def test_boundary_kkt_with_l1_zeros():
    # Strong l1 forces some coordinates to the boundary; check complementarity.
    rng = np.random.default_rng(5)
    D = rng.random((4, 2)) + 0.2
    Y = np.zeros((4, 3)) + 0.01
    M = np.ones_like(Y, dtype=bool)
    W = np.ones_like(Y)
    out = spectral.analyze(D, Y, mask=M, reliability=W, l1=0.2,
                           ridge=0.5, temporal=0.4)
    assert out["converged"]
    g = out["gradient"]
    A = out["activations"]
    # Boundary KKT: A==0 implies grad >= 0 (subgradient stationarity).
    assert np.all(g[A <= 1e-9] >= -1e-7)
    # Interior KKT: grad == 0.
    interior = A > 1e-6
    assert np.all(np.abs(g[interior]) < 1e-7)


def test_temporal_coupling_changes_but_not_objective_sense():
    D = np.array([[1.0], [0.0], [0.0], [0.05]])
    Y = np.array([[1.5, 0.0, 1.5], [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    M = np.ones_like(Y, dtype=bool)
    W = np.ones_like(Y)
    flat = spectral.analyze(D, Y, mask=M, reliability=W,
                            ridge=0.5, temporal=0.0)
    coupled = spectral.analyze(D, Y, mask=M, reliability=W,
                               ridge=0.5, temporal=2.0)
    assert flat["converged"] and coupled["converged"]
    # Coupling smooths the middle frame upward relative to the uncoupled case.
    assert coupled["activations"][0, 1] > flat["activations"][0, 1]
    # Each run is independently optimal for its own objective.
    for out, temp in ((flat, 0.0), (coupled, 2.0)):
        A_ref, _ = lbfsb_minimize(D, Y, M, W, 0.0, 0.5, temp,
                                  np.zeros((D.shape[1], Y.shape[1])))
        assert np.max(np.abs(out["activations"] - A_ref)) < 1e-7


def test_iteration_limit_is_reported_not_success():
    D, Y, M, W = make_case(seed=4)
    out = spectral.analyze(D, Y, mask=M, reliability=W, ridge=0.5,
                           temporal=1.5, max_iter=1)
    assert out["status"] in ("iteration_limit", "converged")
    if not out["converged"]:
        assert out["status"] == "iteration_limit"
        assert out["iterations"] == 1
    # Even a capped run must not be worse than initialization.
    assert out["objective"] <= out["initial_objective"] + 1e-9


def test_initial_is_never_worsened():
    D, Y, M, W = make_case(seed=9, missing=True)
    rng = np.random.default_rng(0)
    A0 = rng.random((D.shape[1], Y.shape[1])) * 2.5
    out = spectral.analyze(D, Y, mask=M, reliability=W, ridge=0.5,
                           temporal=0.8, initial=A0, max_iter=2)
    assert out["objective"] <= out["initial_objective"] + 1e-9


def test_numerical_failure_returns_best_feasible(monkeypatch):
    D, Y, M, W = make_case(seed=1)
    # Force the line search to fail; the documented status must appear and a
    # feasible (nonnegative) point must still come back.
    import spectral as spec
    real = spec._objective_and_gradient
    calls = {"n": 0}

    def flaky(A, *args, **kwargs):
        calls["n"] += 1
        obj, g, c = real(A, *args, **kwargs)
        if calls["n"] > 3:
            return np.nan, np.full_like(g, np.nan), c
        return obj, g, c

    monkeypatch.setattr(spec, "_objective_and_gradient", flaky)
    out = spec.analyze(D, Y, mask=M, reliability=W, ridge=0.5)
    assert out["status"] == "numerical_failure"
    assert out["converged"] is False
    assert np.all(out["activations"] >= 0.0) and np.all(np.isfinite(
        out["activations"]))


def test_inputs_are_not_mutated_and_validated():
    D, Y, M, W = make_case(seed=6)
    Dc, Yc, Mc, Wc = D.copy(), Y.copy(), M.copy(), W.copy()
    spectral.analyze(D, Y, mask=M, reliability=W, ridge=0.5)
    np.testing.assert_array_equal(D, Dc)
    np.testing.assert_array_equal(Y, Yc)
    np.testing.assert_array_equal(M, Mc)
    np.testing.assert_array_equal(W, Wc)

    bad_D = D.copy(); bad_D[0, 0] = -1.0
    with pytest.raises(ValueError):
        spectral.analyze(bad_D, Y, ridge=0.5)
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, ridge=0.05)            # ridge below range
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, l1=0.3)
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, temporal=2.1)
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, reliability=np.full_like(Y, 0.05))
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, mask=np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, max_iter=0)
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, max_iter=1.5)
    bad_init = np.full((D.shape[1], Y.shape[1]), 4.0)
    with pytest.raises(ValueError):
        spectral.analyze(D, Y, initial=bad_init)
    zero_col = D.copy(); zero_col[:, 0] = 0.0
    with pytest.raises(ValueError):
        spectral.analyze(zero_col, Y, ridge=0.5)


def test_gradient_shape_and_defaults():
    D = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.1]])
    Y = np.array([[1.0], [0.5], [0.25], [0.1]])
    out = spectral.analyze(D, Y, ridge=0.5)
    assert out["activations"].shape == (2, 1)
    assert out["gradient"].shape == (2, 1)
    assert out["tolerance"] == 1e-7
    assert out["converged"] and out["residual"] <= 1e-7
