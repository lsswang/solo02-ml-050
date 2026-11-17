"""Joint nonnegative short-time spectral decomposition.

Minimizes, over ``A[P,T] >= 0``, the single global objective

    J(A) = .5 * sum_{f,t} mask[f,t] W[f,t] (sum_p D[f,p] A[p,t] - Y[f,t])^2
           + l1      * sum_{p,t} A[p,t]
           + ridge/2 * sum_{p,t} A[p,t]^2
           + temporal/2 * sum_{p,t>0} (A[p,t] - A[p,t-1])^2 .

The problem is a convex nonnegative quadratic program (ridge > 0 makes it
strictly convex, hence the minimizer is unique).  It is solved jointly over
all frames with accelerated projected gradient (restarted FISTA) carrying a
single full ``A`` state: frames are never solved independently and smoothed
afterwards.  Convergence is decided solely by the projected-gradient
(first-order KKT) residual

    r(A) = max | A - max(A - grad J(A), 0) | ,

and ``r <= 1e-7`` is the only way to obtain ``converged``.  The optimizer's
own success flag is never trusted: the best feasible iterate seen since
initialization is always what is returned.
"""

import numpy as np

_TOL = 1e-7            # documented KKT convergence gate
_SOLVE_TOL = 1e-12     # active-set polish target
_FISTA_GATE = 1e-6     # FISTA hand-off: close enough to identify active sets
_GRAD_ATOL = 1e-8
_GRAD_RTOL = 1e-8


def _as_matrix(name, value, dtype=float):
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    return arr


def _check_finite_nonneg(name, arr, upper=None):
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if upper is not None and np.any(arr > upper):
        raise ValueError(f"{name} elements must be <= {upper}")


def _check_scalar(name, value, low, high):
    val = float(value)
    if not np.isfinite(val) or val < low or val > high:
        raise ValueError(f"{name} must be a finite scalar in [{low}, {high}]")
    return val


def _objective_and_gradient(A, H, Gt, D, M, Y, l1, ridge, temporal):
    """Full objective and full gradient from the precomputed quadratic form."""
    P, T = A.shape
    # H[:, :, t] and Gt[:, t] are the per-frame Hessian and linear term.
    Ha = np.einsum("pqt,qt->pt", H, A)
    grad = Ha - Gt + l1
    if temporal > 0.0 and T > 1:
        # temporal/2 * sum_t (a_t - a_{t-1})^2 with zero outside the grid
        lap = np.empty_like(A)
        lap[:, 0] = A[:, 0] - A[:, 1]
        lap[:, -1] = A[:, -1] - A[:, -2]
        if T > 2:
            lap[:, 1:-1] = 2.0 * A[:, 1:-1] - A[:, :-2] - A[:, 2:]
        grad += temporal * lap

    residual = D @ A - Y
    data = 0.5 * float(np.sum(M * residual * residual))
    l1_cost = l1 * float(np.sum(A))
    ridge_cost = 0.5 * ridge * float(np.sum(A * A))
    if temporal > 0.0 and T > 1:
        deltas = A[:, 1:] - A[:, :-1]
        temporal_cost = 0.5 * temporal * float(np.sum(deltas * deltas))
    else:
        temporal_cost = 0.0
    objective = data + l1_cost + ridge_cost + temporal_cost
    return objective, grad, {
        "data": data,
        "l1": l1_cost,
        "ridge": ridge_cost,
        "temporal": temporal_cost,
    }


def _active_set_polish(A, H, Gt, D, M, Y, l1, ridge, temporal,
                       max_passes=200):
    """Exact-active-set Newton polish on the same nonnegative QP.

    FISTA identifies the active set; the strictly convex smooth QP on the
    current free set is then solved directly (the l1 term is linear, so it
    stays inside the free-coordinate gradient).  Coordinates that the free
    solve drives negative are removed one at a time, and a KKT failure at a
    zero coordinate re-admits it.  All stopping checks use the full gradient
    over the full nonnegative constraint, so the projected-gradient residual
    definition is unchanged.  Returns a ``(P, T)`` matrix and a status string.
    """
    P, T = A.shape
    # Frame-major (Fortran) vectorization: index t*P+p holds A[p, t], so a
    # frame's P pitches are contiguous and temporal neighbors are P apart.
    x = A.flatten(order="F").copy()
    n = x.size

    def full_fg(z):
        obj, g, _ = _objective_and_gradient(
            z.reshape(P, T, order="F"), H, Gt, D, M, Y, l1, ridge, temporal)
        return obj, g.flatten(order="F")

    # Block-tridiagonal Hessian of the smooth part (data + ridge + path).
    # Diagonal frame blocks H_t and off-diagonal -temporal*I between frames.
    K = np.zeros((n, n))
    for t in range(T):
        sl = slice(t * P, (t + 1) * P)
        K[sl, sl] = H[:, :, t]
        if temporal > 0.0 and t > 0:
            prev = slice((t - 1) * P, t * P)
            K[sl, prev] -= temporal * np.eye(P)
            K[prev, sl] -= temporal * np.eye(P)
            K[sl, sl] += temporal * np.eye(P)
            K[prev, prev] += temporal * np.eye(P)

    obj, grad = full_fg(x)
    if not (np.isfinite(obj) and np.all(np.isfinite(grad))):
        return x.reshape(P, T, order="F"), "numerical_failure"

    for _ in range(max_passes):
        free = x > 0.0
        if not np.any(free):
            # All-zero candidate: valid iff every full-gradient entry >= 0.
            if np.all(grad >= -1e-13):
                return np.zeros_like(x).reshape(P, T), "ok"
            x[int(np.argmin(grad))] = 1e-12
            free = x > 0.0
        idx = np.flatnonzero(free)
        Kf = K[np.ix_(free, free)]
        # Free-coordinate stationarity: grad(x) = K x + linear + l1 = 0,
        # so K_free d = -(grad_free - K_free x_free) gives the new point.
        rhs = -(grad[free] - Kf @ x[free])
        try:
            d = np.linalg.solve(Kf, rhs)
        except np.linalg.LinAlgError:
            return x.reshape(P, T, order="F"), "numerical_failure"
        if np.any(~np.isfinite(d)):
            return x.reshape(P, T, order="F"), "numerical_failure"
        if np.any(d < 0.0):
            # Drop the most violated coordinate and repeat.
            x[int(idx[np.argmin(d)])] = 0.0
            obj, grad = full_fg(x)
            continue
        x_new = x.copy()
        x_new[idx] = d
        obj_new, grad_new = full_fg(x_new)
        if not (np.isfinite(obj_new) and np.all(np.isfinite(grad_new))):
            return x.reshape(P, T, order="F"), "numerical_failure"
        # KKT on zero coordinates with the full gradient (l1 included).
        zero = ~free
        if np.any(grad_new[zero] < -1e-11):
            admit = int(np.flatnonzero(zero)[np.argmin(grad_new[zero])])
            x_new[admit] = 1e-12
            x, obj, grad = x_new, obj_new, grad_new
            continue
        x, obj, grad = x_new, obj_new, grad_new
        residual = float(np.max(np.abs(x - np.maximum(x - grad, 0.0))))
        if residual <= _SOLVE_TOL:
            return x.reshape(P, T, order="F"), "ok"
    return x.reshape(P, T, order="F"), "iteration_limit"


def _projected_gradient_solve(initial, H, Gt, D, M, Y, l1, ridge, temporal,
                              max_iter, tol=_FISTA_GATE):
    """Accelerated projected gradient (restarted FISTA), Armijo safeguarded.

    The full matrix ``A`` is the only state: every frame and template is
    optimized jointly.  The residual is always evaluated at the actual
    iterate ``x`` via ``max |x - max(x - grad J(x), 0)|``.  The best feasible
    point seen since initialization is always returned.
    """
    A = initial.copy()
    P, T = A.shape
    # Frame-major (Fortran) ordering, consistent with the active-set polish
    # and the per-frame Hessian block assembly.
    x = A.flatten(order="F")

    def fg(z):
        obj, g, _ = _objective_and_gradient(
            z.reshape(P, T, order="F"), H, Gt, D, M, Y, l1, ridge, temporal)
        return obj, g.flatten(order="F")

    # Lipschitz estimate of grad J (quadratic): largest per-frame Hessian
    # eigenvalue (each frame already includes the ridge term).
    lip = float(np.max([np.linalg.eigvalsh(H[:, :, t])[-1]
                        for t in range(T)]))
    if temporal > 0.0 and T > 1:
        lip += 4.0 * temporal  # largest eigenvalue of the path Laplacian
    lip = max(lip, 1e-8)
    step = lip
    step_ceiling = 1e18
    armijo = 1e-4

    obj, grad = fg(x)
    if not (np.isfinite(obj) and np.all(np.isfinite(grad))):
        return A, "numerical_failure", 0, np.inf

    best_x = x.copy()
    best_obj = obj
    z = x.copy()          # FISTA extrapolation point
    momentum = 1.0
    status = "iteration_limit"
    iterations = 0
    residual = np.inf

    for iterations in range(1, max_iter + 1):
        _, grad_z = fg(z)
        if not np.all(np.isfinite(grad_z)):
            status = "numerical_failure"
            break

        # Backtracking line search on the extrapolated point.
        trial_step = step
        fy = fg(z)[0]
        x_new = np.maximum(z - grad_z / trial_step, 0.0)
        obj_new, grad_new = fg(x_new)
        ok = (np.isfinite(obj_new) and np.all(np.isfinite(grad_new))
              and obj_new <= fy + armijo * float(grad_z @ (x_new - z)))
        if not ok:
            for _ in range(200):
                trial_step *= 2.0  # shorter step
                if trial_step > step_ceiling:
                    break
                x_new = np.maximum(z - grad_z / trial_step, 0.0)
                obj_new, grad_new = fg(x_new)
                if (np.isfinite(obj_new) and np.all(np.isfinite(grad_new))
                        and obj_new <= fy
                        + armijo * float(grad_z @ (x_new - z))):
                    ok = True
                    break
        if not ok:
            status = "numerical_failure"
            break
        step = trial_step

        residual = float(np.max(
            np.abs(x_new - np.maximum(x_new - grad_new, 0.0))))
        if not np.isfinite(residual):
            status = "numerical_failure"
            break

        # Restarted FISTA: a non-monotone step discards momentum, which keeps
        # global convergence for the convex nonnegative quadratic program.
        if obj_new > obj + 1e-15 * (1.0 + abs(obj)):
            momentum = 1.0
            z = x_new.copy()
        else:
            momentum_next = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
            z = np.maximum(
                x_new + ((momentum - 1.0) / momentum_next) * (x_new - x), 0.0)
            momentum = momentum_next

        x, obj, grad = x_new, obj_new, grad_new
        if obj < best_obj:
            best_obj, best_x = obj, x.copy()
        if residual <= tol:
            status = "converged"
            break

    fista_hit_gate = status == "converged"

    if status != "numerical_failure":
        # Exact active-set polish on the same objective. FISTA only has to
        # identify the active set; the polish removes the slow zero-
        # identification tail caused by the nonsmooth l1 term. It is applied
        # to the best feasible point seen within the iteration budget.
        polished, pstatus = _active_set_polish(
            best_x.reshape(P, T, order="F"), H, Gt, D, M, Y,
            l1, ridge, temporal)
        if pstatus == "numerical_failure":
            status = "numerical_failure"
        else:
            x = polished.flatten(order="F")
            _, grad = fg(x)
            residual = float(np.max(
                np.abs(x - np.maximum(x - grad, 0.0))))
            if not np.isfinite(residual):
                status = "numerical_failure"
            elif fista_hit_gate and pstatus == "ok" and residual <= _TOL:
                # Both the projected-gradient budget converged and the exact
                # KKT point satisfies the documented residual gate.
                status = "converged"
            else:
                # The iteration budget was exhausted (or polish could not
                # certify the gate): report the budget status honestly while
                # still returning the best feasible point found.
                status = "iteration_limit"
            best_x = x.copy()
    else:
        _, g_best = fg(best_x)
        residual = float(np.max(
            np.abs(best_x - np.maximum(best_x - g_best, 0.0))))

    A[:] = best_x.reshape(P, T, order="F")
    return A, status, iterations, float(min(residual, np.finfo(float).max))


def analyze(templates, spectrum, mask=None, reliability=None,
            l1=0.0, ridge=0.5, temporal=0.0, initial=None, max_iter=1000):
    """Jointly decompose ``spectrum`` against ``templates``.

    Parameters
    ----------
    templates : array_like, shape (F, P)
        Calibrated nonnegative templates, 1 <= P <= 4, 3 <= F <= 8.
    spectrum : array_like, shape (F, T), 1 <= T <= 8
        Observed nonnegative short-time magnitude/power spectrum.
    mask : array_like of bool, shape (F, T), optional
        False marks a missing bin (not an observed zero); activation variables
        and regularizers at that frame are retained.
    reliability : array_like, shape (F, T), optional
        Finite weights in [0.1, 5]; defaults to ones.
    l1, ridge, temporal : float
        Regularizer strengths in [0, .2], [.1, 1], [0, 2] respectively.
    initial : array_like, shape (P, T), optional
        Finite nonnegative entries <= 3; defaults to the zero matrix.
    max_iter : positive int
        Projected-gradient iteration budget (default 1000).

    Returns
    -------
    dict with ``activations``, ``objective``, per-term ``costs``, the full
    same-shape ``gradient``, KKT ``residual``, ``status``, ``iterations``,
    ``converged`` and the initial objective.
    """
    D = _as_matrix("templates", templates)
    Y = _as_matrix("spectrum", spectrum)
    F, P = D.shape
    if Y.shape[0] != F:
        raise ValueError("templates and spectrum must have the same F")
    T = Y.shape[1]
    if not (1 <= P <= 4 and 3 <= F <= 8 and 1 <= T <= 8):
        raise ValueError("required dimensions are 1<=P<=4, 3<=F<=8, 1<=T<=8")
    _check_finite_nonneg("templates", D, upper=3.0)
    _check_finite_nonneg("spectrum", Y, upper=3.0)
    if np.any(np.sum(D > 0.0, axis=0) == 0):
        raise ValueError("every template column must contain a positive value")

    if mask is None:
        M_bool = np.ones((F, T), dtype=bool)
    else:
        M_bool = np.asarray(mask, dtype=bool)
        if M_bool.shape != (F, T):
            raise ValueError("mask must have the same shape as the spectrum")
    if reliability is None:
        W = np.ones((F, T))
    else:
        W = _as_matrix("reliability", reliability)
        if W.shape != (F, T):
            raise ValueError("reliability must have the same shape as spectrum")
        if not np.all(np.isfinite(W)) or np.any(W < 0.1) or np.any(W > 5.0):
            raise ValueError("reliability entries must be finite in [0.1, 5]")

    l1 = _check_scalar("l1", l1, 0.0, 0.2)
    ridge = _check_scalar("ridge", ridge, 0.1, 1.0)
    temporal = _check_scalar("temporal", temporal, 0.0, 2.0)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) \
            or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    if initial is None:
        A0 = np.zeros((P, T))
    else:
        A0 = _as_matrix("initial", initial)
        if A0.shape != (P, T):
            raise ValueError("initial must have shape (P, T)")
        _check_finite_nonneg("initial", A0, upper=3.0)

    # Quadratic-form assembly: .5 a_t^T H_t a_t - g_t^T a_t per frame, with
    # M = mask * reliability. Missing bins simply get zero weight while the
    # activation variables at that frame stay in the problem.
    M = M_bool.astype(float) * W
    H = np.empty((P, P, T))
    Gt = np.empty((P, T))
    for t in range(T):
        w = M[:, t]
        Dw = D * w[:, None]
        H[:, :, t] = D.T @ Dw + ridge * np.eye(P)
        Gt[:, t] = Dw.T @ Y[:, t]

    initial_objective, _, _ = _objective_and_gradient(
        A0, H, Gt, D, M, Y, l1, ridge, temporal)

    A, status, iterations, residual = _projected_gradient_solve(
        A0, H, Gt, D, M, Y, l1, ridge, temporal, max_iter)

    objective, gradient, costs = _objective_and_gradient(
        A, H, Gt, D, M, Y, l1, ridge, temporal)

    # Safeguard: a returned iterate must never be worse than initialization
    # beyond rounding noise (the solver already tracks the best feasible point).
    if objective > initial_objective + 1e-9:
        A = A0.copy()
        objective, gradient, costs = _objective_and_gradient(
            A, H, Gt, D, M, Y, l1, ridge, temporal)
        residual = float(np.max(np.abs(A - np.maximum(A - gradient, 0.0))))
        status = "numerical_failure"

    # The solver already sets status solely from the projected-gradient
    # residual gate and the iteration budget; never override it here from an
    # optimizer-internal success flag. A capped run that did not earn the
    # documented gate stays ``iteration_limit`` even if its best point is good.
    return {
        "activations": A,
        "objective": float(objective),
        "costs": costs,
        "gradient": gradient,
        "residual": float(residual),
        "tolerance": _TOL,
        "gradient_atol": _GRAD_ATOL,
        "gradient_rtol": _GRAD_RTOL,
        "status": status,
        "converged": status == "converged",
        "iterations": int(iterations),
        "max_iter": int(max_iter),
        "initial_objective": float(initial_objective),
    }
