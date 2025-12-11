"""Benchmark k-means, Gaussian Mixture Models, and vitk on a small suite of datasets.

Kept datasets
-------------
- Heavy-tail synthetic (Student-t clusters)
- Contaminated blobs with 10%, 20%, 50% uniform outliers
- Real toy datasets: iris, seeds, wine, glass, vehicle, banknote

For every dataset it measures runtime, ARI, NMI, within/between-cluster (W/B) ratio,
and mean squared error (MSE) for three algorithms: traditional k-means, GMM, and vitk.

Run examples
------------
python experiments.py                 # synthetic only
python experiments.py --include-real  # add iris + seeds + wine
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import csv

import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.io import arff

from vitk import vitk
from tk import tkmeans


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray
    k: int
    description: str = ""


def make_heavy_tail(
    *,
    n_samples: int = 1600,
    n_features: int = 60,
    centers: int = 4,
    df: float = 2.2,
    df_jitter: float = 0.2,
    spread: float = 8.0,
    anisotropy: float = 1.5,
    cov_scale: float = 0.4,
    random_state: int = 42,
) -> DatasetSpec:
    """Generate clustered heavy-tailed data with anisotropic covariances.

    Each cluster is drawn from a multivariate Student-t with its own random
    covariance and a degrees-of-freedom jittered around ``df`` to avoid
    being too optimistic. Covariance eigenvalues follow a log-normal spread
    controlled by ``anisotropy``.
    """

    rng = np.random.default_rng(random_state)
    means = rng.uniform(-spread, spread, size=(centers, n_features))

    # Equal-size allocation per cluster
    per = n_samples // centers
    leftover = n_samples - per * centers
    counts = [per] * centers
    for i in range(leftover):
        counts[i] += 1

    samples = []
    labels = []
    for k_idx, count in enumerate(counts):
        # cluster-specific df with light jitter
        df_k = max(1.1, df + rng.uniform(-df_jitter, df_jitter))

        # random anisotropic covariance via random orthogonal basis + lognormal eigenvalues
        Q, _ = np.linalg.qr(rng.standard_normal((n_features, n_features)))
        eigs = np.exp(rng.normal(loc=0.0, scale=np.log(anisotropy), size=n_features))
        cov = cov_scale * (Q * eigs) @ Q.T  # PSD with controllable overall scale
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(n_features))

        # Student-t sampling: x = mu + L z / sqrt(chi/df)
        z = rng.standard_normal((count, n_features)) @ L.T
        chi = rng.chisquare(df_k, size=(count, 1))
        scale = np.sqrt(chi / df_k)
        x = means[k_idx] + z / scale
        samples.append(x)
        labels.append(np.full(count, k_idx, dtype=int))

    X = np.vstack(samples)
    y = np.concatenate(labels)
    df_range = f"{df:.2f}±{df_jitter:.2f}"
    desc = (
        f"Student-t (df~{df_range}), anisotropy~{anisotropy}x, "
        f"{n_samples} samples, {n_features} dims, {centers} clusters"
    )
    return DatasetSpec("Heavy-tail t", X, y, centers, desc)


def make_contaminated(
    contamination: float,
    *,
    n_samples: int = 1500,
    n_features: int = 20,
    centers: int = 4,
    cluster_std: float = 0.9,
    spread: float = 12.0,
    random_state: int = 42,
) -> DatasetSpec:
    if not 0 < contamination < 1:
        raise ValueError("contamination must be in (0, 1)")

    n_outliers = int(n_samples * contamination)
    n_inliers = n_samples - n_outliers

    rng = np.random.default_rng(random_state)

    # Sample cluster means uniformly over the hypercube for diversity
    means = rng.uniform(-spread, spread, size=(centers, n_features))

    # Equal-size allocation per cluster (put remainder in the first clusters)
    per = n_inliers // centers
    leftover = n_inliers - per * centers
    counts = [per] * centers
    for i in range(leftover):
        counts[i] += 1

    # Draw inliers from isotropic Gaussians around each mean
    X_in_list = []
    y_in_list = []
    for k_idx, count in enumerate(counts):
        samples_k = rng.normal(loc=means[k_idx], scale=cluster_std, size=(count, n_features))
        X_in_list.append(samples_k)
        y_in_list.append(np.full(count, k_idx, dtype=int))

    X_in = np.vstack(X_in_list)
    y_in = np.concatenate(y_in_list)

    # Uniform outliers across the same hypercube
    outliers = rng.uniform(-spread, spread, size=(n_outliers, n_features))
    X = np.vstack([X_in, outliers])

    # Label outliers as an extra class to evaluate robustness via ARI/NMI.
    y_out = np.full(n_outliers, centers, dtype=int)
    y = np.concatenate([y_in, y_out])

    perc = int(contamination * 100)
    desc = (
        f"{n_samples} samples ({n_inliers} clean + {n_outliers} outliers), "
        f"contamination={perc}%"
    )
    # Use k = centers + 1 to allow a dedicated outlier cluster during fitting.
    return DatasetSpec(f"Outliers {perc}%", X, y, centers + 1, desc)


def build_datasets(random_state: int) -> List[DatasetSpec]:
    datasets: List[DatasetSpec] = [make_heavy_tail(random_state=random_state)]
    for frac in (0.10, 0.20, 0.50):
        datasets.append(make_contaminated(contamination=frac, random_state=random_state))
    return datasets


# ---------------------------------------------------------------------------
# Real-world toy datasets (small, in-memory)
# ---------------------------------------------------------------------------


def load_iris_ds() -> DatasetSpec:
    iris = datasets.load_iris()
    desc = "sklearn iris, 150 samples, 4 dims, 3 classes"
    return DatasetSpec("Iris", iris.data, iris.target, k=len(np.unique(iris.target)), description=desc)


def load_seeds_ds(data_home: str | None = "./openml_cache") -> DatasetSpec:
    """Load UCI 'seeds' dataset via OpenML. Small; cached by sklearn if available.

    If offline or unavailable, raises RuntimeError so caller can skip gracefully.
    """

    try:
        seeds = datasets.fetch_openml(
            "seeds", version=1, as_frame=False, parser="liac-arff", data_home=data_home
        )
    except Exception as exc:  # network/offline friendly
        raise RuntimeError(f"Could not load 'seeds' dataset: {exc}") from exc

    X = seeds.data
    y = seeds.target.astype(int) - 1  # labels are 1,2,3
    desc = "UCI seeds, 210 samples, 7 dims, 3 classes"
    return DatasetSpec("Seeds", X, y, k=len(np.unique(y)), description=desc)


def load_wine_ds() -> DatasetSpec:
    wine = datasets.load_wine()
    desc = "sklearn wine, 178 samples, 13 dims, 3 classes"
    return DatasetSpec("Wine", wine.data, wine.target, k=len(np.unique(wine.target)), description=desc)


def _encode_labels(y: np.ndarray) -> np.ndarray:
    """Map arbitrary labels to 0..C-1 preserving order of appearance."""

    uniq, inverse = np.unique(y, return_inverse=True)
    # ensure deterministic order using the sorted unique already returned by np.unique
    return inverse.astype(int)


def load_glass_ds(data_home: str | None = "./openml_cache") -> DatasetSpec:
    try:
        glass = datasets.fetch_openml("glass", version=1, as_frame=False, parser="liac-arff", data_home=data_home)
    except Exception as exc:
        raise RuntimeError(f"Could not load 'glass' dataset: {exc}") from exc

    X = glass.data.astype(np.float64)
    y = _encode_labels(glass.target)
    desc = "UCI glass identification, 214 samples, 9 dims, 6 classes"
    return DatasetSpec("Glass", X, y, k=len(np.unique(y)), description=desc)


def load_vehicle_ds(data_home: str | None = "./openml_cache") -> DatasetSpec:
    try:
        vehicle = datasets.fetch_openml("vehicle", version=1, as_frame=False, parser="liac-arff", data_home=data_home)
    except Exception as exc:
        raise RuntimeError(f"Could not load 'vehicle' dataset: {exc}") from exc

    X = vehicle.data.astype(np.float64)
    y = _encode_labels(vehicle.target)
    desc = "UCI vehicle silhouettes, 846 samples, 18 dims, 4 classes"
    return DatasetSpec("Vehicle", X, y, k=len(np.unique(y)), description=desc)


def load_banknote_ds(data_home: str | None = "./openml_cache") -> DatasetSpec:
    try:
        bank = datasets.fetch_openml(
            "banknote-authentication", version=1, as_frame=False, parser="liac-arff", data_home=data_home
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load 'banknote-authentication' dataset: {exc}") from exc

    X = bank.data.astype(np.float64)
    y = _encode_labels(bank.target)
    desc = "Banknote authentication, 1,372 samples, 4 dims, 2 classes"
    return DatasetSpec("Banknote", X, y, k=len(np.unique(y)), description=desc)


# ---------------------------------------------------------------------------
# ARFF datasets living under ./datasets
# ---------------------------------------------------------------------------


ARFF_MAX_SAMPLES_DEFAULT = 20_000


def _to_float(value: object) -> float:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return float("nan")
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _decode_label(value: object) -> object:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value
    return value


def load_arff_dataset(path: Path, *, max_samples: int = ARFF_MAX_SAMPLES_DEFAULT) -> DatasetSpec:
    data, meta = arff.loadarff(path)
    n_samples = data.shape[0]
    n_features = len(meta.names()) - 1

    if n_samples > max_samples:
        raise RuntimeError(f"{n_samples} samples exceed limit {max_samples}")

    raw = np.asarray(data.tolist(), dtype=object)
    X_list = [[_to_float(v) for v in row[:-1]] for row in raw]
    y_list = [_decode_label(row[-1]) for row in raw]

    X = np.asarray(X_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=object)

    # Drop rows with missing features to keep models happy
    mask = ~np.isnan(X).any(axis=1)
    if not np.all(mask):
        dropped = int(mask.size - int(np.sum(mask)))
        X = X[mask]
        y_arr = y_arr[mask]
        n_samples = X.shape[0]
        desc_drop = f", dropped {dropped} rows with missing values"
    else:
        desc_drop = ""

    y = _encode_labels(y_arr)
    k = len(np.unique(y))
    if k < 2:
        raise RuntimeError("dataset has fewer than 2 unique labels")
    desc = f"{path.parent.name}, {n_samples} samples, {n_features} dims{desc_drop}"
    return DatasetSpec(path.stem, X, y, k, description=desc)


def discover_arff_datasets(root: Path, *, max_samples: int) -> List[Tuple[DatasetSpec, str]]:
    if not root.exists():
        return []

    datasets: List[Tuple[DatasetSpec, str]] = []
    for arff_path in sorted(root.glob("**/*.arff")):
        subset = arff_path.parent.name
        try:
            spec = load_arff_dataset(arff_path, max_samples=max_samples)
        except Exception as exc:  # skip oversized or malformed datasets
            print(f"[warn] skip {arff_path.name}: {exc}")
            continue
        datasets.append((spec, subset))
    return datasets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mse_and_wb(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    assigned = centers[labels]
    diff = X - assigned
    within = float(np.sum(diff * diff))
    mse = within / X.shape[0]

    counts = np.bincount(labels, minlength=centers.shape[0]).astype(float)
    global_mean = np.mean(X, axis=0)
    center_shift = centers - global_mean
    between = float(np.sum(counts[:, None] * center_shift * center_shift))
    wb_ratio = within / between if between > 0 else float("inf")
    return mse, wb_ratio


def format_float(value: float, width: int = 10, precision: int = 4) -> str:
    if np.isnan(value):
        return f"{'nan':>{width}}"
    if np.isinf(value):
        return f"{'inf':>{width}}"
    return f"{value:{width}.{precision}f}"


# ---------------------------------------------------------------------------
# Algorithm wrappers
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    model: str
    ari: float
    nmi: float
    mse: float
    wb: float
    runtime: float
    notes: str = ""


def run_vitk(
    X: np.ndarray,
    y_true: np.ndarray,
    k: int,
    seed: int,
    *,
    nu: float | None = None,
    nu_max: float | None = None,
    nu_damping: float | None = None,
    nu_update_every: int | None = None,
    alpha_dir_prior: float | None = None,
    beta0: float | None = None,
    a0: float | None = None,
    tau_temperature: float | None = None,
    max_iter: int | None = None,
    patience: int | None = None,
    tol: float | None = None,
) -> ModelResult:
    vitk_kwargs = {
        "k": k,
        "random_state": seed,
        "init": "kmeans",
        "verbose": False,
    }
    # Only override if provided to keep vitk's tuned defaults
    if nu is not None:
        vitk_kwargs["nu"] = nu
    if nu_max is not None:
        vitk_kwargs["nu_max"] = nu_max
    if nu_damping is not None:
        vitk_kwargs["nu_damping"] = nu_damping
    if nu_update_every is not None:
        vitk_kwargs["nu_update_every"] = nu_update_every
    if alpha_dir_prior is not None:
        vitk_kwargs["alpha_dir_prior"] = alpha_dir_prior
    if beta0 is not None:
        vitk_kwargs["beta0"] = beta0
    if a0 is not None:
        vitk_kwargs["a0"] = a0
    if tau_temperature is not None:
        vitk_kwargs["tau_temperature"] = tau_temperature
    if max_iter is not None:
        vitk_kwargs["max_iter"] = max_iter
    if patience is not None:
        vitk_kwargs["patience"] = patience
    if tol is not None:
        vitk_kwargs["tol"] = tol

    start = time.perf_counter()
    labels_vi, centers_vi, _resp_vi, stats_vi = vitk(X, **vitk_kwargs)
    runtime = time.perf_counter() - start
    labels_vi_zero = labels_vi - 1
    ari = adjusted_rand_score(y_true, labels_vi_zero)
    nmi = normalized_mutual_info_score(y_true, labels_vi_zero)
    mse, wb = mse_and_wb(X, centers_vi, labels_vi_zero)
    notes = f"iters={stats_vi['iterations']}"
    return ModelResult("vitk", ari, nmi, mse, wb, runtime, notes)


def run_kmeans(X: np.ndarray, y_true: np.ndarray, k: int, seed: int) -> ModelResult:
    model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    start = time.perf_counter()
    labels = model.fit_predict(X)
    runtime = time.perf_counter() - start
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    mse, wb = mse_and_wb(X, model.cluster_centers_, labels)
    notes = f"n_iter={getattr(model, 'n_iter_', 'n/a')}"
    return ModelResult("k-means", ari, nmi, mse, wb, runtime, notes)


def run_gmm(X: np.ndarray, y_true: np.ndarray, k: int, seed: int) -> ModelResult:
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=1,
        random_state=seed,
        init_params="kmeans",
    )
    start = time.perf_counter()
    model.fit(X)
    labels = model.predict(X)
    runtime = time.perf_counter() - start
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    mse, wb = mse_and_wb(X, model.means_, labels)
    notes = f"converged={model.converged_}, n_iter={model.n_iter_}"
    return ModelResult("GMM", ari, nmi, mse, wb, runtime, notes)


def run_all_models(
    X: np.ndarray,
    y_true: np.ndarray,
    k: int,
    seed: int,
    vitk_overrides: dict | None = None,
) -> List[ModelResult]:
    vitk_overrides = vitk_overrides or {}
    return [
        run_kmeans(X, y_true, k, seed),
        run_gmm(X, y_true, k, seed),
        run_vitk(X, y_true, k, seed, **vitk_overrides),
        run_tk(X, y_true, k),
    ]


def run_tk(X: np.ndarray, y_true: np.ndarray, k: int) -> ModelResult:
    start = time.perf_counter()
    labels_one_based, centers, iters = tkmeans(
        X,
        k=k,
        nu_fixed=5.0,
        max_iter=800,
        tol=1e-3,
        patience=20,
    )
    runtime = time.perf_counter() - start
    labels = labels_one_based - 1  # convert to 0-based for metrics
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    mse, wb = mse_and_wb(X, centers, labels)
    notes = f"iters={iters}"
    return ModelResult("tkmeans", ari, nmi, mse, wb, runtime, notes)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def print_results(dataset: DatasetSpec, results: Iterable[ModelResult]) -> None:
    header = (
        f"{'Model':<18}{'ARI':>10}{'NMI':>10}{'MSE':>14}{'W/B':>14}{'Time(s)':>10}  Notes"
    )
    print(f"\nDataset: {dataset.name} | {dataset.description} | k={dataset.k}")
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.model:<18}"
            f"{format_float(res.ari):>10}"
            f"{format_float(res.nmi):>10}"
            f"{format_float(res.mse, 14, 6)}"
            f"{format_float(res.wb, 14, 6)}"
            f"{res.runtime:10.3f}  {res.notes}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare k-means, GMM, and vitk on synthetic datasets; optional small real datasets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--scale",
        choices=["standard", "robust", "none"],
        default=None,
        help="Feature scaling strategy. Default: robust for heavy-tail dataset, standard otherwise.",
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        help="Append small real datasets (iris, seeds, wine, glass, vehicle, banknote) to the benchmark run.",
    )
    parser.add_argument(
        "--use-datasets-folder",
        action="store_true",
        help="Benchmark every .arff file under ./datasets (artificial + real-world). Large datasets are skipped automatically.",
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default="./datasets",
        help="Root directory that contains 'artificial' and 'real-world' folders with .arff datasets.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=ARFF_MAX_SAMPLES_DEFAULT,
        help="Skip .arff datasets that have more rows than this threshold.",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="./csv",
        help="Directory to store the aggregated CSV results.",
    )
    parser.add_argument("--vitk-nu", type=float, default=None, help="Override vitk nu")
    parser.add_argument("--vitk-nu-max", type=float, default=None, help="Override vitk nu_max")
    parser.add_argument("--vitk-nu-damping", type=float, default=None, help="Override vitk nu_damping")
    parser.add_argument("--vitk-nu-update-every", type=int, default=None, help="Override vitk nu_update_every")
    parser.add_argument("--vitk-alpha", type=float, default=None, help="Override vitk alpha_dir_prior")
    parser.add_argument("--vitk-beta0", type=float, default=None, help="Override vitk beta0")
    parser.add_argument("--vitk-a0", type=float, default=None, help="Override vitk a0")
    parser.add_argument("--vitk-tau-temp", type=float, default=None, help="Override vitk tau_temperature")
    parser.add_argument("--vitk-max-iter", type=int, default=None, help="Override vitk max_iter")
    parser.add_argument("--vitk-patience", type=int, default=None, help="Override vitk patience")
    parser.add_argument("--vitk-tol", type=float, default=None, help="Override vitk tol")
    args = parser.parse_args()

    datasets: List[Tuple[DatasetSpec, str]] = []
    datasets.extend((ds, "synthetic") for ds in build_datasets(args.seed))

    if args.include_real:
        datasets.append((load_iris_ds(), "toy-real"))
        try:
            datasets.append((load_seeds_ds(), "toy-real"))
        except RuntimeError as exc:
            print(f"[warn] skip seeds dataset: {exc}")
        datasets.append((load_wine_ds(), "toy-real"))
        for loader in (load_glass_ds, load_vehicle_ds, load_banknote_ds):
            try:
                datasets.append((loader(), "toy-real"))
            except RuntimeError as exc:
                print(f"[warn] skip {loader.__name__[5:-3]} dataset: {exc}")

    if args.use_datasets_folder:
        datasets.extend(
            discover_arff_datasets(Path(args.datasets_root), max_samples=args.max_samples)
        )

    if not datasets:
        print("No datasets to run. Enable --use-datasets-folder or --include-real.")
        return

    vitk_overrides = {
        "nu": args.vitk_nu,
        "nu_max": args.vitk_nu_max,
        "nu_damping": args.vitk_nu_damping,
        "nu_update_every": args.vitk_nu_update_every,
        "alpha_dir_prior": args.vitk_alpha,
        "beta0": args.vitk_beta0,
        "a0": args.vitk_a0,
        "tau_temperature": args.vitk_tau_temp,
        "max_iter": args.vitk_max_iter,
        "patience": args.vitk_patience,
        "tol": args.vitk_tol,
    }

    csv_rows: List[dict] = []

    for ds, origin in datasets:
        scale_choice = args.scale
        if scale_choice is None:
            scale_choice = "robust" if ds.name == "Heavy-tail t" else "standard"

        if scale_choice == "standard":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(ds.X)
        elif scale_choice == "robust":
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(ds.X)
        else:
            X_scaled = ds.X

        try:
            results = run_all_models(X_scaled, ds.y, ds.k, args.seed, vitk_overrides)
        except Exception as exc:
            print(f"[warn] skip {ds.name}: {exc}")
            continue

        print_results(ds, results)

        for res in results:
            csv_rows.append(
                {
                    "dataset": ds.name,
                    "origin": origin,
                    "description": ds.description,
                    "n_samples": ds.X.shape[0],
                    "n_features": ds.X.shape[1],
                    "k": ds.k,
                    "model": res.model,
                    "ari": res.ari,
                    "nmi": res.nmi,
                    "mse": res.mse,
                    "wb": res.wb,
                    "runtime": res.runtime,
                    "notes": res.notes,
                }
            )

    if csv_rows:
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = csv_dir / f"experiments_{timestamp}.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[info] wrote {len(csv_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
