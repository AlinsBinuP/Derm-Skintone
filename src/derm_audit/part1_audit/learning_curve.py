"""Sample-size-to-performance projection (manuscript section 3.3).

Fits the saturating exponential  f(n) = a [1 - exp(-n / b)]  to anchor points
measured by training on random class-balanced subsets, then reads the curve at
the median adequately-sampled size of each skin-tone band.

Reproduces manuscript table 3 and table 4.
"""


from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

__all__ = [
    "saturating_exponential",
    "inverse_power_law",
    "CurveFit",
    "fit_curve",
    "aic",
    "project_gap",
    "sensitivity_table",
]


def saturating_exponential(n, a, b):
    """f(n) = a [1 - exp(-n / b)] — manuscript equation (1)."""
    return a * (1.0 - np.exp(-np.asarray(n, dtype=float) / b))


def inverse_power_law(n, a, c, d):
    """f(n) = a - c * n**(-d) — the comparison form in section 3.3."""
    return a - c * np.power(np.asarray(n, dtype=float), -d)


@dataclass
class CurveFit:
    """A fitted learning curve with parameter uncertainty."""

    params: np.ndarray
    cov: np.ndarray
    residuals: np.ndarray
    func: callable
    names: tuple = field(default=("a", "b"))

    @property
    def stderr(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def ci(self, level: float = 0.95, dof: int | None = None) -> dict:
        """Two-sided t confidence intervals for each parameter."""
        if dof is None:
            dof = max(len(self.residuals) - len(self.params), 1)
        t = stats.t.ppf(0.5 + level / 2.0, dof)
        return {
            name: (p - t * se, p + t * se)
            for name, p, se in zip(self.names, self.params, self.stderr)
        }

    def __call__(self, n):
        return self.func(n, *self.params)

    @property
    def r_squared(self) -> float:
        y = self.residuals + self(self._n) if hasattr(self, "_n") else None
        if y is None:
            return float("nan")
        ss_res = float(np.sum(self.residuals**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_curve(n, f, model: str = "exponential", p0=None) -> CurveFit:
    """Fit a learning curve by non-linear least squares.

    Parameters
    ----------
    n : array of training images per class
    f : array of attained macro-F1 at each size
    model : "exponential" (manuscript equation 1) or "power" (inverse power law)
    """
    n = np.asarray(n, dtype=float)
    f = np.asarray(f, dtype=float)
    if n.size != f.size:
        raise ValueError("n and f must have the same length")
    if n.size < 3:
        raise ValueError("at least three anchor points are required")

    if model == "exponential":
        func, names = saturating_exponential, ("a", "b")
        p0 = p0 or [float(f.max()), float(np.median(n))]
    elif model == "power":
        func, names = inverse_power_law, ("a", "c", "d")
        p0 = p0 or [float(f.max()) * 1.02, 2.0, 0.6]
    else:
        raise ValueError(f"unknown model {model!r}")

    params, cov = curve_fit(func, n, f, p0=p0, maxfev=40000)
    residuals = f - func(n, *params)
    cf = CurveFit(params=params, cov=cov, residuals=residuals, func=func, names=names)
    cf._n = n
    return cf


def aic(residuals, n_params: int) -> float:
    """Akaike information criterion for a least-squares fit."""
    residuals = np.asarray(residuals, dtype=float)
    m = residuals.size
    rss = float(np.sum(residuals**2))
    if rss <= 0:
        rss = np.finfo(float).tiny
    return m * np.log(rss / m) + 2 * n_params


def project_gap(
    cf: CurveFit,
    n_lighter: float,
    n_darker: float,
    n_boot: int = 2000,
    seed: int = 0,
    anchors=None,
) -> dict:
    """Project the macro-F1 gap between the two bands, with a bootstrap interval.

    The interval is obtained by resampling the anchor residuals, refitting the
    curve on each resample and taking the 2.5th and 97.5th percentiles
    (manuscript section 3.3).

    Note
    ----
    If the anchor points lie almost exactly on the fitted curve the residual
    spread is near zero and the interval will be implausibly tight. Report the
    anchors at full precision so that the interval reflects real run-to-run
    variation.
    """
    f_lo, f_hi = float(cf(n_lighter)), float(cf(n_darker))
    point = f_lo - f_hi

    n = anchors if anchors is not None else getattr(cf, "_n", None)
    if n is None:
        return {"gap": point, "lighter": f_lo, "darker": f_hi, "ci": (np.nan, np.nan)}

    rng = np.random.default_rng(seed)
    fitted = cf(n)
    draws = []
    for _ in range(n_boot):
        resampled = fitted + rng.choice(cf.residuals, size=len(n), replace=True)
        try:
            p, _ = curve_fit(cf.func, n, resampled, p0=cf.params, maxfev=40000)
        except Exception:
            continue
        draws.append(cf.func(n_lighter, *p) - cf.func(n_darker, *p))

    draws = np.asarray(draws, dtype=float)
    ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    return {
        "gap": point,
        "lighter": f_lo,
        "darker": f_hi,
        "ci": ci,
        "boot_mean": float(draws.mean()) if draws.size else float("nan"),
        "n_boot_ok": int(draws.size),
    }


def sensitivity_table(anchors_n, anchors_f, settings, n_boot: int = 2000, seed: int = 0):
    """Build manuscript table 4.

    `settings` is a sequence of (label, model, n_lighter, n_darker) tuples.
    """
    rows = []
    for label, model, n_lo, n_hi in settings:
        cf = fit_curve(anchors_n, anchors_f, model=model)
        res = project_gap(cf, n_lo, n_hi, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "setting": label,
                "model": model,
                "n_lighter": n_lo,
                "n_darker": n_hi,
                "gap": round(res["gap"], 3),
                "ci_low": round(res["ci"][0], 3),
                "ci_high": round(res["ci"][1], 3),
                "aic": round(aic(cf.residuals, len(cf.params)), 2),
            }
        )
    return rows
