"""Variational-inference-based Student-t k-means (vitk).

This module implements a variational Bayes analogue of the tkmeans routine.
It uses a Dirichlet prior over mixture weights, a Normal-Gamma prior over
cluster means / precisions, and latent Gamma scale variables to recover the
heavy-tailed Student-t likelihood. The algorithm alternates closed-form
variational updates until the responsibilities converge.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.special import digamma, polygamma
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score


logger = logging.getLogger(__name__)


def _softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical stability."""
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    exp_logits_sum = exp_logits.sum(axis=1, keepdims=True)
    exp_logits_sum = np.clip(exp_logits_sum, 1e-12, None)
    return exp_logits / exp_logits_sum


def vitk(
    data_set,
    k: int,
    nu: float = 5,
    learn_nu: bool = True,
    nu_max_iter: int = 40,
    nu_update_every: int = 2,
    nu_damping: float = 0.3,
    nu_max: float | None = 50.0,
    max_iter: int = 300,
    tol: float = 5e-4,
    patience: int = 8,
    elbo_tol: float | None = None,
    alpha_dir_prior: float = 1.5,
    beta0: float = 0.08,
    a0: float | None = None,
    b0: float = 1.0,
    random_state: int | None = 42,
    init: str = "kmeans",
    verbose: bool = True,
    standardize: bool = False,
    scaler: str | None = None,
    transpose_if_wide: bool = True,
    tau_temperature: float = 1.0,
    dtype: str | np.dtype = np.float64,
):
    """Variational Student-t k-means.

    Parameters
    ----------
    data_set : array-like, shape (n_samples, n_features)
        Input data matrix.
    k : int
        Number of clusters.
    nu : float, optional
        Degrees of freedom for the Student-t heavy-tailed component (initial value when ``learn_nu=True``).
    learn_nu : bool, optional
        If ``True`` update ``nu`` each iteration via damped Newton steps on the VB fixed-point equation.
    nu_max_iter : int, optional
        Maximum Newton iterations per outer loop when updating ``nu``.
    nu_update_every : int, optional
        Perform ``nu`` update every N outer iterations to improve stability on noisy data.
    nu_damping : float, optional
        Multiplicative damping factor for the Newton step (<=1.0 for stability).
    nu_max : float, optional
        Upper bound for ``nu`` to avoid runaway updates; ``None`` disables the cap.
    max_iter : int, optional
        Maximum number of variational updates.
    tol : float, optional
        Convergence threshold on responsibilities.
    patience : int, optional
        Number of consecutive iterations satisfying ``tol`` required to stop.
    elbo_tol : float, optional
        Optional plateau threshold on the (approximate) ELBO; when set, the loop also
        stops after ``patience`` rounds with |ΔELBO| < ``elbo_tol``.
    alpha_dir_prior : float, optional
        Symmetric Dirichlet prior concentration for mixture weights.
    beta0 : float, optional
        Strength of the Normal component in the Normal-Gamma prior.
    a0, b0 : float, optional
        Shape / rate of the Gamma prior over precisions. If ``a0`` is ``None``
        it defaults to ``1 + p / 4`` (milder than the previous 1 + p/2) for heavy-tail robustness.
    random_state : int, optional
        Seed for reproducibility.
    init : {'random', 'kmeans'}, optional
        Initialization strategy for responsibilities/centers. ``'kmeans'`` warm-starts
        with sklearn KMeans (more stable in高维数据); ``'random'`` keeps the original
        random simplex initialization.
    verbose : bool, optional
        If ``True`` prints diagnostic information each iteration.
    standardize : bool, optional
        Backwards-compatible flag. Ignored if ``scaler`` is provided.
    scaler : {'none','standard','robust'}, optional
        Choose feature scaling strategy. ``robust`` is often better for heavy tails. ``None`` falls back to ``standardize``.
    transpose_if_wide : bool, optional
        If ``True`` and n_samples < n_features, transpose to (n_samples, n_features).
    tau_temperature : float, optional
        Softmax temperature (>1 makes assignments softer; helps avoid early collapse).
    dtype : np.dtype or str, optional
        Force computation dtype (e.g., ``np.float32`` to save memory).

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Hard cluster assignments in ``[1, k]``.
    mean_posterior : ndarray, shape (k, p)
        Posterior mean of every cluster center.
    responsibilities : ndarray, shape (n_samples, k)
        Final responsibilities (variational posteriors over labels).
    stats : dict
        Additional diagnostics (iterations, Dirichlet params, etc.).
    """

    # legacy 'standardize' kept for backwards compatibility; 'scaler' takes precedence
    if scaler is None:
        scaler = "standard" if standardize else "none"
    scaler = scaler.lower()
    if scaler not in {"none", "standard", "robust"}:
        raise ValueError("scaler must be one of {'none','standard','robust'} or None")

    X = np.array(data_set, dtype=dtype)
    if X.ndim != 2:
        raise ValueError("data_set must be a 2D array")
    if transpose_if_wide and X.shape[0] < X.shape[1]:
        X = X.T
    fitted_scaler: Optional[StandardScaler] = None
    if scaler == "standard":
        fitted_scaler = StandardScaler()
        X = fitted_scaler.fit_transform(X)
    elif scaler == "robust":
        from sklearn.preprocessing import RobustScaler

        fitted_scaler = RobustScaler()
        X = fitted_scaler.fit_transform(X)
    n, p = X.shape
    g = int(k)
    if g <= 1:
        raise ValueError("k must be >= 2 for clustering")

    a0 = (1.0 + 0.25 * p) if a0 is None else float(a0)
    if a0 <= 1.0:
        raise ValueError("a0 must be > 1.0 for finite E[1/alpha]")

    if verbose and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(random_state)
    if init not in {"random", "kmeans"}:
        raise ValueError("init must be 'random' or 'kmeans'")

    if init == "kmeans":
        # Warm-start with k-means for better separation in高维场景。
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=g, n_init=5, random_state=random_state)
        labels0 = km.fit_predict(X)
        tau = np.zeros((n, g))
        tau[np.arange(n), labels0] = 1.0
        mean_posterior = km.cluster_centers_.astype(float)
    else:
        tau = rng.random((n, g))
        tau /= np.clip(tau.sum(axis=1, keepdims=True), 1e-12, None)

        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        mean_posterior = rng.uniform(xmin, xmax, size=(g, p))
    m0 = np.mean(X, axis=0)

    lambda_shape = 0.5 * (nu + p)
    dirichlet_alpha = np.ones(g) * alpha_dir_prior + tau.sum(axis=0)
    E_log_pi = digamma(dirichlet_alpha) - digamma(dirichlet_alpha.sum())

    beta_k = np.ones(g) * (beta0 + n / g)
    a_k = np.ones(g) * (a0 + 0.5 * p)
    b_k = np.ones(g) * (b0 + 1.0)

    stable_counter = 0
    elbo_stable_counter = 0
    last_max_change = np.inf
    last_elbo: float | None = None
    elbo_history: list[float] = []

    for itr in range(1, max_iter + 1):
        E_alpha = a_k / b_k
        E_log_alpha = digamma(a_k) - np.log(b_k)
        E_inv_alpha = b_k / (a_k - 1.0 + 1e-9)
        mu_var_term = p * E_inv_alpha / beta_k

        # Compute squared distances without forming full (n, k, p) tensor
        X_norm2 = np.einsum("ij,ij->i", X, X)
        mean_norm2 = np.einsum("ij,ij->i", mean_posterior, mean_posterior)
        cross = X @ mean_posterior.T
        sq_dist = np.clip(X_norm2[:, None] - 2.0 * cross + mean_norm2[None, :], 0.0, None)
        E_quad = sq_dist + mu_var_term[None, :]

        lambda_rate = 0.5 * (nu + E_alpha[None, :] * E_quad)
        E_lambda = lambda_shape / lambda_rate
        E_log_lambda = digamma(lambda_shape) - np.log(lambda_rate)

        log_tau = (
            E_log_pi[None, :]
            + 0.5 * p * (E_log_alpha[None, :] + E_log_lambda - np.log(2 * np.pi))
            - 0.5 * E_alpha[None, :] * E_lambda * E_quad
        )
        if tau_temperature <= 0:
            raise ValueError("tau_temperature must be positive")
        tau_new = _softmax_stable(log_tau / tau_temperature)
        max_change = np.max(np.abs(tau_new - tau))
        tau = tau_new

        # Track an approximate ELBO for optional convergence checks / diagnostics
        log_tau_norm = np.log(np.clip(tau, 1e-12, None))
        elbo = float(np.sum(tau * (log_tau - log_tau_norm)))
        elbo_history.append(elbo)

        Nk = tau.sum(axis=0)
        dirichlet_alpha = alpha_dir_prior + Nk
        E_log_pi = digamma(dirichlet_alpha) - digamma(np.sum(dirichlet_alpha))

        weights = tau * E_lambda
        Nk_tilde = weights.sum(axis=0) + 1e-12
        beta_k = beta0 + Nk_tilde

        weighted_means = weights.T @ X
        x_bar = np.divide(
            weighted_means,
            Nk_tilde[:, None],
            out=np.tile(m0, (g, 1)),
            where=Nk_tilde[:, None] > 1e-12,
        )
        mean_posterior = (beta0 * m0 + weighted_means) / beta_k[:, None]

        # scatter per cluster without materializing (n, k, p)
        normX2 = np.einsum("ij,ij->i", X, X)
        scatter = weights.T @ normX2 - Nk_tilde * np.einsum("ij,ij->i", x_bar, x_bar)
        mean_shift = np.sum((x_bar - m0) ** 2, axis=1)

        a_k = np.maximum(a0 + 0.5 * Nk_tilde * p, 1.0 + 1e-6)
        b_k = b0 + 0.5 * scatter + 0.5 * beta0 * Nk_tilde / beta_k * mean_shift

        # Optional update for degrees of freedom nu via damped Newton steps
        if learn_nu and (itr % max(nu_update_every, 1) == 0):
            # Average term in fixed-point equation f(nu)=0 from VB derivation
            avg_term = np.sum(tau * (E_log_lambda - E_lambda)) / (n * g)
            for _ in range(nu_max_iter):
                f_val = np.log(nu / 2.0) - digamma(nu / 2.0) + 1.0 - avg_term
                f_prime = (1.0 / nu) - 0.5 * polygamma(1, nu / 2.0)
                step = f_val / (f_prime + 1e-12)
                step *= nu_damping
                nu_new = nu - step
                if nu_max is not None:
                    nu_new = min(nu_new, nu_max)
                nu_new = max(nu_new, 2.05)
                if abs(step) < 1e-4:
                    nu = nu_new
                    break
                nu = nu_new
            # refresh lambda_shape for next outer iteration
            lambda_shape = 0.5 * (nu + p)
            if verbose and nu_max is not None and abs(nu - nu_max) < 1e-8:
                logger.info("nu reached cap (%.1f); consider lowering prior strength or cap", nu_max)

        if verbose:
            logger.info(
                "itr: %3d, max|Δτ|: %.3e, minNk: %.2f, maxNk: %.2f, nu: %.2f, ELBO: %.3e",
                itr,
                max_change,
                Nk.min(),
                Nk.max(),
                nu,
                elbo,
            )

        if max_change < tol or abs(max_change - last_max_change) < tol:
            stable_counter += 1
        else:
            stable_counter = 0

        if elbo_tol is not None and last_elbo is not None:
            if abs(elbo - last_elbo) < elbo_tol:
                elbo_stable_counter += 1
            else:
                elbo_stable_counter = 0

        if stable_counter >= patience or (elbo_tol is not None and elbo_stable_counter >= patience):
            if verbose:
                logger.info("Converged after %d iterations.", itr)
            break
        last_max_change = max_change
        last_elbo = elbo

    labels = np.argmax(tau, axis=1) + 1
    stats = {
        "iterations": itr,
        "dirichlet_alpha": dirichlet_alpha,
        "beta": beta_k,
        "a": a_k,
        "b": b_k,
        "responsibilities": tau,
        "nu": nu,
        "elbo_history": elbo_history,
        "scaler": fitted_scaler,
        "scaler_type": scaler,
    }
    stats["centers_scaled"] = mean_posterior
    centers_out = mean_posterior
    if fitted_scaler is not None:
        centers_out = fitted_scaler.inverse_transform(mean_posterior)
    return labels, centers_out, tau, stats


class VariationalStudentKMeans:
    """Sklearn-like estimator wrapper for ``vitk``.

    Minimal API supporting ``fit`` and ``predict`` so it can be dropped into
    simple pipelines without changing the core algorithm implementation.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        nu: float = 1.5,
        learn_nu: bool = True,
        standardize: bool = False,
        random_state: int | None = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.nu = nu
        self.learn_nu = learn_nu
        self.standardize = standardize
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X, y=None):
        labels, centers, resp, stats = vitk(
            X,
            k=self.n_clusters,
            nu=self.nu,
            learn_nu=self.learn_nu,
            standardize=self.standardize,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.kwargs,
        )
        self.labels_ = labels - 1  # switch to 0-based to match sklearn convention
        self.cluster_centers_ = centers
        self.responsibilities_ = resp
        self.stats_ = stats
        self.nu_ = stats.get("nu", self.nu)
        self.n_iter_ = stats.get("iterations")
        self._centers_scaled = stats.get("centers_scaled")
        self._scaler = stats.get("scaler")
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Call fit before predict.")
        X_arr = np.array(X, dtype=np.float64)
        centers = self._centers_scaled if self._scaler is not None else self.cluster_centers_
        if self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)
        centers_mat = np.array(centers, dtype=X_arr.dtype)
        x_norm2 = np.einsum("ij,ij->i", X_arr, X_arr)
        c_norm2 = np.einsum("ij,ij->i", centers_mat, centers_mat)
        cross = X_arr @ centers_mat.T
        sq = np.clip(x_norm2[:, None] - 2 * cross + c_norm2[None, :], 0.0, None)
        return np.argmin(sq, axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    iris = datasets.load_iris()
    X = iris.data
    y_true = iris.target + 1


    labels_vi, centers_vi, resp_vi, stats_vi = vitk(
        X,
        k=3,
        # nu=15.0,
        max_iter=300,
        tol=5e-4,
        patience=5,
        verbose=True,
    )

    ari = adjusted_rand_score(y_true, labels_vi)
    sil = silhouette_score(X, labels_vi)
    print(f"ARI: {ari:.4f}, Silhouette: {sil:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_vi, s=20, cmap="viridis")
    plt.scatter(centers_vi[:, 0], centers_vi[:, 1], c="red", s=200, marker="X")
    plt.title(f"Variational t-k-means (iters={stats_vi['iterations']})")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.tight_layout()
    plt.show()
