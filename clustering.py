"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Satellite Image Clustering — EM (GMM) vs Fuzzy C-Means  v5            ║
║  Interactive Analysis Tool  |  GNR605 Assignment                        ║
║  Input: Satellite Image (GeoTIFF / PNG)                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

pip install rasterio numpy scikit-learn matplotlib tqdm scipy
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────
import sys, os, warnings, time
import numpy as np
import matplotlib
try:
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as _rio_from_bounds
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MAX_PIXELS        = 80_000
CHUNK             = 8_000
METRICS_SUBSAMPLE = 10_000
SILHOUETTE_SAMPLE = 3_000
RANDOM_SEED       = 42

# Default parameters (hidden from UI — sensible fixed values)
DEFAULT_K            = 4
DEFAULT_SAMPLE_FRAC  = 0.40
DEFAULT_GMM_COV      = "full"
DEFAULT_GMM_N_INIT   = 5
DEFAULT_GMM_MAX_ITER = 300      # fixed, not exposed
DEFAULT_GMM_REG      = 1e-6     # fixed, not exposed
DEFAULT_FCM_M        = 2.0
DEFAULT_FCM_MAX_ITER = 150      # fixed, not exposed
DEFAULT_FCM_TOL      = 1e-4     # fixed, not exposed
DEFAULT_FCM_INIT     = "kmeans++"

PALETTE = np.array([
    [0.08, 0.40, 0.75],
    [0.11, 0.37, 0.13],
    [0.73, 0.11, 0.11],
    [0.46, 0.76, 0.26],
    [0.83, 0.71, 0.51],
    [0.90, 0.55, 0.10],
    [0.50, 0.20, 0.70],
    [0.95, 0.95, 0.30],
])
CLASS_NAMES = ["Water", "Dense Veg", "Urban", "Sparse Veg",
               "Bare/Road", "Mixed", "Shadow", "Other"]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FUZZY C-MEANS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class FuzzyCMeans:
    def __init__(self, n_clusters=4, m=2.0, max_iter=150,
                 tol=1e-4, random_state=RANDOM_SEED, init="kmeans++"):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.centers_ = None
        self.u_ = None
        self.n_iter_ = 0
        self.inertia_ = np.inf

    def _init_kpp(self, X, rng):
        idx = rng.integers(0, X.shape[0])
        centers = [X[idx]]
        for _ in range(1, self.n_clusters):
            dists = np.min(
                np.array([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0)
            probs = dists / (dists.sum() + 1e-12)
            centers.append(X[rng.choice(X.shape[0], p=probs)])
        return np.array(centers)

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n, _ = X.shape
        c, m = self.n_clusters, self.m

        if self.init == "kmeans++":
            v = self._init_kpp(X, rng)
            d0 = np.maximum(
                np.linalg.norm(X[np.newaxis] - v[:, np.newaxis], axis=2), 1e-10)
            inv0 = (1.0 / d0) ** (2.0 / (m - 1))
            u = inv0 / inv0.sum(axis=0, keepdims=True)
        else:
            u = rng.random((c, n))
            u /= u.sum(axis=0, keepdims=True)

        d = np.ones((c, n))
        for it in range(self.max_iter):
            u_old = u.copy()
            um = u ** m
            v  = (um @ X) / um.sum(axis=1, keepdims=True)
            d  = np.maximum(
                np.linalg.norm(X[np.newaxis] - v[:, np.newaxis], axis=2), 1e-10)
            inv = (1.0 / d) ** (2.0 / (m - 1))
            u   = inv / inv.sum(axis=0, keepdims=True)
            self.n_iter_ = it + 1
            if np.max(np.abs(u - u_old)) < self.tol:
                break

        self.u_ = u
        self.centers_ = v
        self.labels_ = np.argmax(u, axis=0)
        self.inertia_ = float(np.sum((u ** m) * d ** 2))
        return self

    def predict_proba(self, X):
        """Unified membership formula — (n_samples, n_clusters)."""
        dist = np.maximum(
            np.linalg.norm(
                X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :], axis=2),
            1e-10)
        inv = (1.0 / dist) ** (2.0 / (self.m - 1))
        return inv / inv.sum(axis=1, keepdims=True)

    @property
    def partition_coefficient(self):
        return float(np.sum(self.u_ ** 2) / self.u_.shape[1])

    @property
    def partition_entropy(self):
        return float(-np.sum(self.u_ * np.log(self.u_ + 1e-10)) / self.u_.shape[1])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMAGE LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def load_image(path, max_dim=800):
    print(f"\n  Loading: {path}")
    with rasterio.open(path) as src:
        n_bands = src.count
        H, W    = src.height, src.width
        crs     = src.crs
        bounds  = src.bounds
        print(f"  Size  : {H}×{W}  bands={n_bands}  CRS={crs}")
        scale   = min(max_dim / max(H, W), 1.0)
        new_H, new_W = int(H * scale), int(W * scale)
        bands = [src.read(b, out_shape=(new_H, new_W),
                          resampling=Resampling.bilinear).astype(np.float32)
                 for b in range(1, n_bands + 1)]

    img = np.stack(bands, axis=-1)
    print(f"  Loaded: {img.shape}  (scale={scale:.2f})")

    img_norm = np.zeros_like(img)
    for b in range(img.shape[2]):
        ch = img[:, :, b]
        p2, p98 = (np.percentile(ch[ch > 0], [2, 98])
                   if np.any(ch > 0) else (0.0, 1.0))
        img_norm[:, :, b] = np.clip((ch - p2) / (p98 - p2 + 1e-6), 0, 1)

    transform = (_rio_from_bounds(bounds.left, bounds.bottom,
                                  bounds.right, bounds.top, new_W, new_H)
                 if bounds is not None else None)
    return img_norm, (H, W, n_bands), (new_H, new_W), transform, crs


def build_feature_matrix(img):
    H, W, B = img.shape
    X     = img.reshape(-1, B).astype(np.float32)
    valid = np.all(np.isfinite(X) & (X >= 0) & (X <= 1), axis=1)
    return X, valid


def compute_rgb_display(img, gamma=1.3):
    H, W, B = img.shape
    if B >= 3:
        rgb = img[:, :, :3]
    elif B == 2:
        rgb = np.stack([img[:, :, 0], img[:, :, 1], img[:, :, 0]], axis=-1)
    else:
        rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
    return np.clip(np.power(rgb, 1.0 / gamma), 0, 1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLUSTERING ENGINE                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def run_clustering(X, valid, H, W, n_clusters, algorithm,
                   fuzz_m=DEFAULT_FCM_M,
                   fcm_init=DEFAULT_FCM_INIT,
                   gmm_cov_type=DEFAULT_GMM_COV,
                   gmm_n_init=DEFAULT_GMM_N_INIT,
                   sample_frac=DEFAULT_SAMPLE_FRAC):
    n_total  = int(np.sum(valid))
    n_sample = max(min(int(n_total * sample_frac), MAX_PIXELS),
                   min(n_total, 5000))
    rng       = np.random.default_rng(RANDOM_SEED)
    valid_idx = np.where(valid)[0]
    samp_idx  = rng.choice(valid_idx, size=n_sample, replace=False)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X[samp_idx])

    print(f"\n  Fitting {algorithm}  k={n_clusters}  "
          f"on {n_sample:,} pixels …", end=" ", flush=True)
    t0 = time.time()

    if algorithm == "EM (GMM)":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=gmm_cov_type,
            max_iter=DEFAULT_GMM_MAX_ITER,
            n_init=gmm_n_init,
            reg_covar=DEFAULT_GMM_REG,
            random_state=RANDOM_SEED)
        model.fit(X_scaled)
        n_iter = model.n_iter_
    else:
        model = FuzzyCMeans(
            n_clusters=n_clusters, m=fuzz_m,
            max_iter=DEFAULT_FCM_MAX_ITER,
            tol=DEFAULT_FCM_TOL,
            init=fcm_init,
            random_state=RANDOM_SEED)
        model.fit(X_scaled)
        n_iter = model.n_iter_

    print(f"done ({time.time()-t0:.1f}s  iters={n_iter})")
    print(f"  Assigning labels to {n_total:,} pixels …", end=" ", flush=True)

    X_all    = X[valid_idx]
    raw_lbl  = np.empty(len(X_all), dtype=np.int32)
    for cs in tqdm(range(0, len(X_all), CHUNK),
                   desc=f"  [{algorithm}]", ncols=70):
        chunk = scaler.transform(X_all[cs:cs + CHUNK])
        if algorithm == "EM (GMM)":
            raw_lbl[cs:cs + CHUNK] = model.predict(chunk)
        else:
            raw_lbl[cs:cs + CHUNK] = np.argmax(
                model.predict_proba(chunk), axis=1)

    labels_flat = np.full(H * W, -1, dtype=np.int32)
    labels_flat[valid_idx] = raw_lbl
    t_total = time.time() - t0
    print(f"done  ({t_total:.1f}s total)")
    return labels_flat.reshape(H, W), model, scaler, t_total, n_iter


def compute_metrics(X, valid, labels_2d, algorithm, model, scaler):
    labels_flat = labels_2d.ravel()
    valid_mask  = valid & (labels_flat >= 0)
    idx         = np.where(valid_mask)[0]
    rng = np.random.default_rng(RANDOM_SEED)
    if len(idx) > METRICS_SUBSAMPLE:
        idx = rng.choice(idx, METRICS_SUBSAMPLE, replace=False)

    X_sub = X[idx]
    L_sub = labels_flat[idx]
    if len(np.unique(L_sub)) < 2:
        return {}

    metrics = {}
    try:
        metrics["Silhouette Score"]        = silhouette_score(
            X_sub, L_sub, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_SEED)
        metrics["Davies-Bouldin Index"]    = davies_bouldin_score(X_sub, L_sub)
        metrics["Calinski-Harabász Index"] = calinski_harabasz_score(X_sub, L_sub)
    except Exception as e:
        print(f"  [metrics] {e}")

    if algorithm == "EM (GMM)" and model is not None and scaler is not None:
        try:
            metrics["BIC"] = model.bic(scaler.transform(X_sub))
            metrics["AIC"] = model.aic(scaler.transform(X_sub))
        except Exception as e:
            print(f"  [BIC/AIC] {e}")

    if algorithm == "Fuzzy C-Means":
        metrics["Partition Coefficient"] = model.partition_coefficient
        metrics["Partition Entropy"]     = model.partition_entropy

    return metrics


def align_labels(labels_ref, labels_other, k):
    mask = (labels_ref >= 0) & (labels_other >= 0)
    if not np.any(mask):
        return labels_other
    lr, lo = labels_ref[mask], labels_other[mask]
    overlap = np.zeros((k, k), dtype=np.int64)
    np.add.at(overlap, (lr, lo), 1)
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = np.zeros(k, dtype=np.int32)
    for r, c in zip(row_ind, col_ind):
        mapping[c] = r
    new_labels = np.full_like(labels_other, -1)
    v = labels_other >= 0
    new_labels[v] = mapping[labels_other[v]]
    return new_labels


def labels_to_rgb(labels, palette, n_classes):
    pal_ext = np.vstack([palette[:n_classes], [[0.0, 0.0, 0.0]]])
    safe    = np.where(labels < 0, n_classes, labels)
    return pal_ext[safe].astype(np.float32)


def _winner(ev, fv, higher_is_better):
    if ev is None and fv is None: return "—"
    if ev is None: return "FCM ✓"
    if fv is None: return "EM ✓"
    if abs(ev - fv) < 1e-9: return "Tie"
    return "EM ✓" if (ev > fv) == higher_is_better else "FCM ✓"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FILE SELECTION DIALOG                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def select_image_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select Satellite Image",
        filetypes=[("Image files", "*.tif *.tiff *.TIF *.TIFF *.png *.PNG"),
                   ("GeoTIFF", "*.tif *.tiff"), ("PNG", "*.png *.PNG"),
                   ("All files", "*.*")])
    root.destroy()
    return path if path else None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SAVE  (PNG cluster maps + metrics CSV + algo description)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def save_all_results(chosen_dir, labels_em, labels_fcm, k,
                     time_em, time_fcm, n_iter_em, n_iter_fcm,
                     metrics_em, metrics_fcm, X, valid,
                     rgb_display, cluster_names, run_params,
                     fig_results=None):
    """
    Exports:
      • results_<ts>.png          — dashboard screenshot
      • original_rgb_<ts>.png     — input image
      • clusters_EM_<ts>.png      — EM cluster-colour map
      • clusters_FCM_<ts>.png     — FCM cluster-colour map
      • labels_EM_<ts>.npy        — raw integer label array
      • labels_FCM_<ts>.npy       — raw integer label array
      • metrics_<ts>.csv          — quality metrics + per-cluster stats
      • algorithm_description_<ts>.txt
    """
    import datetime
    os.makedirs(chosen_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%d%m_%H%M%S")

    # ── Dashboard screenshot ───────────────────────────────────────────────
    if fig_results is not None:
        p = os.path.join(chosen_dir, f"results_{ts}.png")
        fig_results.savefig(p, dpi=150, bbox_inches="tight",
                            facecolor=fig_results.get_facecolor())
        print(f"\n  ✓ Results PNG        → {p}")

    # ── Original RGB image ─────────────────────────────────────────────────
    if rgb_display is not None:
        rgb_path = os.path.join(chosen_dir, f"original_rgb_{ts}.png")
        plt.imsave(rgb_path, np.clip(rgb_display, 0, 1))
        print(f"  ✓ Original RGB PNG   → {rgb_path}")

    # ── Cluster colour maps as PNG ─────────────────────────────────────────
    for arr, tag in [(labels_em, "EM"), (labels_fcm, "FCM")]:
        if arr is None:
            continue
        # Coloured cluster map
        rgb_map = labels_to_rgb(arr, PALETTE, k)
        cmap_path = os.path.join(chosen_dir, f"clusters_{tag}_{ts}.png")
        plt.imsave(cmap_path, rgb_map)
        print(f"  ✓ Cluster PNG ({tag})     → {cmap_path}")
        # Raw integer label array
        np.save(os.path.join(chosen_dir, f"labels_{tag}_{ts}.npy"), arr)

    # ── Metrics CSV ────────────────────────────────────────────────────────
    rp  = run_params or {}
    csv = os.path.join(chosen_dir, f"metrics_{ts}.csv")
    with open(csv, "w") as f:
        f.write("=== Run Parameters ===\n")
        for key in ["image", "algorithm", "k", "sample_pct",
                    "gmm_cov_type", "gmm_n_init",
                    "fuzz_m", "fcm_init"]:
            f.write(f"{key},{rp.get(key,'')}\n")
        f.write(f"saved_at,{ts}\n")

        names = cluster_names or [f"Cluster {i+1}" for i in range(k)]
        f.write("\n=== Cluster Names ===\n")
        for ci in range(k):
            f.write(f"C{ci+1},{names[ci]}\n")

        f.write("\n=== Quality Metrics ===\n")
        f.write("Metric,EM,FCM\n")
        all_keys = list(dict.fromkeys(
            list(metrics_em.keys()) + list(metrics_fcm.keys())))
        for mk in all_keys:
            f.write(f"{mk},{metrics_em.get(mk,'')},{metrics_fcm.get(mk,'')}\n")
        f.write(f"Time_s,{time_em:.3f},{time_fcm:.3f}\n")
        f.write(f"Iterations,{n_iter_em},{n_iter_fcm}\n")

        if X is not None and valid is not None:
            n_bands = X.shape[1]
            f.write("\n=== Per-Cluster Spectral Statistics ===\n")
            f.write("Algorithm,Cluster," +
                    ",".join(f"Band{b+1}_mean" for b in range(n_bands)) +
                    ",Pixel_count\n")
            for arr, tag in [(labels_em, "EM"), (labels_fcm, "FCM")]:
                if arr is None:
                    continue
                lf = arr.ravel()
                for ci in range(k):
                    mask_c = valid & (lf == ci)
                    if not np.any(mask_c):
                        f.write(f"{tag},C{ci+1}," +
                                ",".join(["—"] * n_bands) + ",0\n")
                    else:
                        means = X[mask_c].mean(axis=0)
                        f.write(f"{tag},C{ci+1}," +
                                ",".join(f"{m:.6f}" for m in means) +
                                f",{int(np.sum(mask_c))}\n")
    print(f"  ✓ Metrics CSV        → {csv}")

    _save_algo_desc(chosen_dir, ts, rp)
    return ts


def _save_algo_desc(output_dir, timestamp, rp=None):
    rp = rp or {}
    fname = os.path.join(output_dir, f"algorithm_description_{timestamp}.txt")
    txt = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SATELLITE IMAGE CLUSTERING — ALGORITHM DESCRIPTION  GNR605             ║
║  Generated: {timestamp}
╚══════════════════════════════════════════════════════════════════════════╝

━━━ 1. EM / GMM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Covariance type : {rp.get('gmm_cov_type', DEFAULT_GMM_COV)}
n_init          : {rp.get('gmm_n_init', DEFAULT_GMM_N_INIT)}
max_iter        : {DEFAULT_GMM_MAX_ITER}  (fixed)
reg_covar       : {DEFAULT_GMM_REG}  (fixed)

E-Step : r(i,k) = πₖ·N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ·N(xᵢ|μⱼ,Σⱼ)
M-Step : update μₖ, Σₖ, πₖ via weighted statistics
Model selection: BIC / AIC (↓ better)

━━━ 2. Fuzzy C-Means ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fuzziness m     : {rp.get('fuzz_m', DEFAULT_FCM_M)}
Init method     : {rp.get('fcm_init', DEFAULT_FCM_INIT)}
max_iter        : {DEFAULT_FCM_MAX_ITER}  (fixed)
tolerance       : {DEFAULT_FCM_TOL}  (fixed)

Centre update : vₖ = Σᵢ (uᵢₖ)ᵐ xᵢ / Σᵢ (uᵢₖ)ᵐ
Membership    : uᵢₖ = 1 / Σⱼ (dᵢₖ/dᵢⱼ)^(2/(m−1))
Quality       : Partition Coefficient (↑), Partition Entropy (↓)

━━━ 3. Run Parameters ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image       : {rp.get('image', '—')}
Algorithm   : {rp.get('algorithm', '—')}
k           : {rp.get('k', '—')}
Sample (%)  : {rp.get('sample_pct', '—')}
""".strip()
    with open(fname, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"  ✓ Algo description   → {fname}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS POPUP WINDOW                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def open_results_window(rgb_display, labels_em, labels_fcm, k,
                        time_em, time_fcm, n_iter_em, n_iter_fcm,
                        metrics_em, metrics_fcm, img, cbar_ref,
                        output_dir=None, run_params=None,
                        fcm_model=None, fcm_scaler=None,
                        X=None, valid=None):
    cmap = ListedColormap([PALETTE[c] for c in range(k)])
    norm = BoundaryNorm(range(k + 1), cmap.N)

    fig = plt.figure(figsize=(20, 11), facecolor="#0D1117")
    fig.canvas.manager.set_window_title("Clustering Results  |  GNR605")
    fig.suptitle(f"Clustering Results  ·  k={k}  ·  "
                 f"EM ({time_em:.1f}s, {n_iter_em} iters)  vs  "
                 f"FCM ({time_fcm:.1f}s, {n_iter_fcm} iters)",
                 color="white", fontsize=13, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.04, right=0.97, top=0.94, bottom=0.27,
                           hspace=0.35, wspace=0.25)

    def _mk(r, c):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#161B22")
        for sp in ax.spines.values(): sp.set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=7)
        return ax

    ax_rgb  = _mk(0, 0)
    ax_em   = _mk(0, 1)
    ax_fcm  = _mk(0, 2)
    ax_diff = _mk(1, 0)
    ax_area = _mk(1, 1)
    ax_met  = _mk(1, 2)

    ax_rgb.imshow(np.clip(rgb_display, 0, 1), interpolation="bilinear")
    ax_rgb.set_title("Original Image (RGB)", color="#C9D1D9", fontsize=10, pad=5)
    ax_rgb.axis("off")

    if labels_em is not None:
        im_em = ax_em.imshow(labels_em, cmap=cmap, norm=norm, interpolation="nearest")
        cb_em = plt.colorbar(im_em, ax=ax_em, fraction=0.04, pad=0.02,
                             ticks=np.arange(k) + 0.5)
        cb_em.ax.set_yticklabels([f"C{i+1}" for i in range(k)],
                                  color="#8B949E", fontsize=7)
        cb_em.ax.tick_params(size=0)
        ax_em.set_title(f"EM / GMM  (k={k},  {time_em:.1f}s,  {n_iter_em} iters)",
                        color="#58A6FF", fontsize=10, pad=5)
    else:
        ax_em.text(0.5, 0.5, "EM not run", transform=ax_em.transAxes,
                   ha="center", color="#58A6FF", fontsize=11)
    ax_em.axis("off")

    if labels_fcm is not None:
        im_fcm = ax_fcm.imshow(labels_fcm, cmap=cmap, norm=norm, interpolation="nearest")
        cb_fcm = plt.colorbar(im_fcm, ax=ax_fcm, fraction=0.04, pad=0.02,
                              ticks=np.arange(k) + 0.5)
        cb_fcm.ax.set_yticklabels([f"C{i+1}" for i in range(k)],
                                   color="#8B949E", fontsize=7)
        cb_fcm.ax.tick_params(size=0)
        ax_fcm.set_title(f"Fuzzy C-Means  (k={k},  {time_fcm:.1f}s,  {n_iter_fcm} iters)",
                         color="#F78166", fontsize=10, pad=5)
    else:
        ax_fcm.text(0.5, 0.5, "FCM not run", transform=ax_fcm.transAxes,
                    ha="center", color="#F78166", fontsize=11)
    ax_fcm.axis("off")

    if labels_em is not None and labels_fcm is not None:
        diff  = (labels_em != labels_fcm).astype(float)
        diff[labels_em < 0] = np.nan
        agree = np.nanmean(labels_em == labels_fcm) * 100
        im_d  = ax_diff.imshow(diff, cmap="RdYlGn_r", vmin=0, vmax=1)
        cb_d  = plt.colorbar(im_d, ax=ax_diff, fraction=0.04, pad=0.02)
        cb_d.ax.set_yticklabels(["Agree", "", "Disagree"],
                                 color="#8B949E", fontsize=7)
        cb_d.ax.tick_params(size=0)
        cbar_ref[0] = cb_d
        ax_diff.set_title(f"EM vs FCM Disagreement  —  Agreement: {agree:.1f}%",
                          color="#D2A8FF", fontsize=10, pad=5)
    elif labels_em is not None:
        ax_diff.imshow(labels_em, cmap=cmap, norm=norm, interpolation="nearest")
        ax_diff.set_title("EM Clustering (single algo)", color="#D2A8FF", fontsize=10)
    elif labels_fcm is not None:
        ax_diff.imshow(labels_fcm, cmap=cmap, norm=norm, interpolation="nearest")
        ax_diff.set_title("FCM Clustering (single algo)", color="#D2A8FF", fontsize=10)
    ax_diff.axis("off")

    has_em  = labels_em  is not None
    has_fcm = labels_fcm is not None
    x, w    = np.arange(k), 0.35
    if has_em:
        counts_em = [int(np.sum(labels_em == c)) for c in range(k)]
        off = -w / 2 if has_fcm else 0
        for ci in range(k):
            ax_area.bar(ci + off, counts_em[ci], w,
                        color=PALETTE[ci % len(PALETTE)], alpha=0.92,
                        edgecolor="#58A6FF", linewidth=1.2,
                        label="EM (GMM)" if ci == 0 else "")
    if has_fcm:
        counts_fcm = [int(np.sum(labels_fcm == c)) for c in range(k)]
        off = w / 2 if has_em else 0
        for ci in range(k):
            ax_area.bar(ci + off, counts_fcm[ci], w,
                        color=PALETTE[ci % len(PALETTE)], alpha=0.60,
                        edgecolor="#F78166", linewidth=1.2,
                        label="FCM" if ci == 0 else "")
    ax_area.set_xticks(x)
    ax_area.set_xticklabels([f"C{c+1}" for c in range(k)],
                            rotation=20, ha="right", color="#8B949E", fontsize=8)
    ax_area.set_ylabel("Pixel Count", color="#8B949E", fontsize=9)
    ax_area.tick_params(axis="y", colors="#8B949E")
    handles = []
    if has_em:
        handles.append(Line2D([0],[0], color="#58A6FF", lw=3, label="EM (GMM)"))
    if has_fcm:
        handles.append(Line2D([0],[0], color="#F78166", lw=3, label="FCM"))
    if handles:
        ax_area.legend(handles=handles, fontsize=7.5, labelcolor="white",
                       framealpha=0.2, facecolor="#161B22", edgecolor="#30363D")
    ax_area.set_title("Cluster Pixel Distribution", color="white", fontsize=10, pad=5)
    ax_area.set_facecolor("#161B22")
    for sp in ax_area.spines.values(): sp.set_color("#30363D")

    ax_met.axis("off")
    ax_met.set_title("Quality Metrics", color="white", fontsize=10, pad=5)
    HB = {"Silhouette Score": True, "Davies-Bouldin Index": False,
          "Calinski-Harabász Index": True, "BIC": False, "AIC": False,
          "Partition Coefficient": True, "Partition Entropy": False}
    rows, rcols = [], []
    for mk in HB:
        ev, fv = metrics_em.get(mk), metrics_fcm.get(mk)
        if ev is None and fv is None:
            continue
        rows.append([mk[:17],
                     f"{ev:.4f}" if ev is not None else "—",
                     f"{fv:.4f}" if fv is not None else "—",
                     _winner(ev, fv, HB[mk])])
        rcols.append(["#1C2128", "#0D1117", "#0D1117", "#1C2128"])
    te = time_em  if time_em  > 1e-6 else None
    tf = time_fcm if time_fcm > 1e-6 else None
    rows.append(["Time (s)",
                 f"{time_em:.2f}" if te else "—",
                 f"{time_fcm:.2f}" if tf else "—",
                 _winner(te, tf, False)])
    rcols.append(["#1C2128", "#0D1117", "#0D1117", "#1C2128"])
    if rows:
        t_tbl = ax_met.table(cellText=rows,
                             colLabels=["Metric", "EM", "FCM", "Better"],
                             cellLoc="center", loc="center",
                             bbox=[0, 0, 1, 0.96])
        t_tbl.auto_set_font_size(False); t_tbl.set_fontsize(8.5)
        for (r, c), cell in t_tbl.get_celld().items():
            cell.set_edgecolor("#30363D")
            if r == 0:
                cell.set_facecolor("#21262D")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(rcols[r - 1][c])
                if c == 1:   cell.set_text_props(color="#58A6FF")
                elif c == 2: cell.set_text_props(color="#F78166")
                elif c == 3: cell.set_text_props(color="#3FB950", fontweight="bold")
                else:        cell.set_text_props(color="#8B949E")

    # Cluster rename text boxes
    cluster_names = [f"Cluster {i+1}" for i in range(k)]
    fig.text(0.04, 0.235, "✏  Rename Clusters (press Enter to confirm):",
             color="#C9D1D9", fontsize=8, fontweight="bold", va="center")
    _tb_refs = []
    bw, x0 = 0.90 / k, 0.05

    def _make_name_cb(idx):
        def _cb(text, _i=idx):
            cluster_names[_i] = text.strip() or f"Cluster {_i+1}"
            _refresh_legend()
        return _cb

    for ci in range(k):
        r2, g2, b2 = PALETTE[ci]
        hx = "#{:02X}{:02X}{:02X}".format(int(r2*255), int(g2*255), int(b2*255))
        lax = fig.add_axes([x0+ci*bw, 0.202, bw*0.30, 0.028])
        lax.set_facecolor(hx); lax.axis("off")
        lax.text(0.5, 0.5, f"C{ci+1}", color="white", fontsize=7,
                 ha="center", va="center", fontweight="bold")
        tax = fig.add_axes([x0+ci*bw+bw*0.31, 0.202, bw*0.65, 0.028])
        for sp in tax.spines.values():
            sp.set_color(hx); sp.set_linewidth(1.5)
        tb = TextBox(tax, "", initial=cluster_names[ci],
                     color="#21262D", hovercolor="#2D333B", label_pad=0.01)
        tb.text_disp.set_color("white"); tb.text_disp.set_fontsize(7.5)
        tb.on_submit(_make_name_cb(ci))
        _tb_refs.append(tb)
        fig.__dict__[f"_tb_{ci}"] = tb

    _legend_holder = [None]

    def _refresh_legend():
        if _legend_holder[0] is not None:
            try: _legend_holder[0].remove()
            except Exception: pass
        patches = [mpatches.Patch(color=PALETTE[c], label=cluster_names[c])
                   for c in range(k)]
        _legend_holder[0] = fig.legend(
            handles=patches, loc="lower center", ncol=k,
            framealpha=0.2, labelcolor="white", fontsize=9,
            bbox_to_anchor=(0.5, 0.155),
            facecolor="#161B22", edgecolor="#30363D")
        fig.canvas.draw_idle()

    _refresh_legend()

    # FCM membership maps button
    ax_mem  = fig.add_axes([0.04, 0.01, 0.28, 0.045])
    btn_mem = Button(ax_mem, "🔬  FCM MEMBERSHIP MAPS",
                     color="#1F3A5F", hovercolor="#264F7A")
    btn_mem.label.set_color("#58A6FF"); btn_mem.label.set_fontsize(9)
    btn_mem.label.set_fontweight("bold")
    fig._btn_mem = btn_mem

    def _show_membership(_):
        if fcm_model is None or fcm_scaler is None or X is None or valid is None:
            save_status.set_text("⚠  FCM not run — no membership data.")
            save_status.set_color("#F78166"); fig.canvas.draw(); return
        save_status.set_text("⏳  Computing membership maps …")
        save_status.set_color("#58A6FF"); fig.canvas.draw(); fig.canvas.flush_events()
        vi  = np.where(valid)[0]
        u   = fcm_model.predict_proba(fcm_scaler.transform(X[vi]))
        ref = labels_fcm if labels_fcm is not None else labels_em
        Hm, Wm = ref.shape
        cols = min(k, 4); rows = (k + cols - 1) // cols
        fm = plt.figure(figsize=(5*cols, 4.5*rows), facecolor="#0D1117")
        fm.canvas.manager.set_window_title("FCM Soft Membership Maps")
        fm.suptitle("FCM Soft Membership Maps  —  per-cluster pixel probability",
                    color="white", fontsize=11, fontweight="bold")
        for ci in range(k):
            am = fm.add_subplot(rows, cols, ci+1)
            am.set_facecolor("#161B22")
            mf = np.full(Hm*Wm, np.nan, dtype=np.float32)
            mf[vi] = u[:, ci]
            im_m = am.imshow(mf.reshape(Hm, Wm), cmap="viridis",
                             vmin=0, vmax=1, interpolation="bilinear")
            cb = plt.colorbar(im_m, ax=am, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors="#8B949E", labelsize=7)
            r2, g2, b2 = PALETTE[ci]
            hx = "#{:02X}{:02X}{:02X}".format(int(r2*255), int(g2*255), int(b2*255))
            am.set_title(f"{cluster_names[ci]}  (C{ci+1})",
                         color=hx, fontsize=10, pad=4)
            am.axis("off")
        fm.tight_layout(rect=[0, 0, 1, 0.95]); fm.show(); plt.pause(0.05)
        save_status.set_text("✓  Membership maps opened.")
        save_status.set_color("#3FB950"); fig.canvas.draw()

    btn_mem.on_clicked(_show_membership)

    # Save button inside results window
    ax_sv = fig.add_axes([0.38, 0.01, 0.24, 0.045])
    btn_sv = Button(ax_sv, "💾  SAVE RESULTS", color="#238636", hovercolor="#2EA043")
    btn_sv.label.set_color("white"); btn_sv.label.set_fontsize(10)
    btn_sv.label.set_fontweight("bold")
    fig._btn_sv = btn_sv

    save_status = fig.text(0.64, 0.028, "", ha="left", va="center",
                           color="#3FB950", fontsize=9, fontfamily="monospace")

    def _do_save(_):
        import tkinter as tk
        from tkinter import filedialog
        rt = tk.Tk(); rt.withdraw(); rt.attributes("-topmost", True)
        d = filedialog.askdirectory(
            title="Select folder to save",
            initialdir=output_dir or os.path.expanduser("~"))
        rt.destroy()
        if not d:
            save_status.set_text("⚠  Cancelled."); save_status.set_color("#F78166")
            fig.canvas.draw(); return
        save_status.set_text("⏳  Saving …"); save_status.set_color("#58A6FF")
        fig.canvas.draw(); fig.canvas.flush_events()
        ts = save_all_results(d, labels_em, labels_fcm, k,
                              time_em, time_fcm, n_iter_em, n_iter_fcm,
                              metrics_em, metrics_fcm, X, valid,
                              rgb_display, cluster_names, run_params,
                              fig_results=fig)
        save_status.set_text(f"✓  Saved [{ts}] → {os.path.basename(d)}/")
        save_status.set_color("#3FB950"); fig.canvas.draw()

    btn_sv.on_clicked(_do_save)
    fig.canvas.draw(); fig.show(); plt.pause(0.05)
    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN CONTROL WINDOW                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class ClusteringApp:
    """
    App launches directly — no startup dialog.
    Image panel shows a placeholder until the user clicks Browse.
    Control panel has two sections:
      Section 1 : Algorithm, k, Sample %
      Section 2 : GMM params (left) | FCM params (right)
    """
    CP_L = 0.570
    CP_R = 0.990
    CP_W = CP_R - CP_L   # ≈ 0.420

    def __init__(self, img=None, orig_size=None, display_size=None,
                 image_path=None, output_dir=None,
                 transform=None, crs=None):
        self.img        = img
        self.orig_size  = orig_size
        self.display_size = display_size
        self.H          = display_size[0] if display_size else 0
        self.W          = display_size[1] if display_size else 0
        self.image_path = image_path
        self.output_dir = output_dir
        self.transform  = transform
        self.crs        = crs

        # Common parameters
        self.n_clusters  = DEFAULT_K
        self.algorithm   = "Both"
        self.sample_frac = DEFAULT_SAMPLE_FRAC
        # GMM parameters (exposed)
        self.gmm_cov_type = DEFAULT_GMM_COV
        self.gmm_n_init   = DEFAULT_GMM_N_INIT
        # FCM parameters (exposed)
        self.fuzz_m   = DEFAULT_FCM_M
        self.fcm_init = DEFAULT_FCM_INIT
        # Results state
        self.labels_em   = None;  self.labels_fcm  = None
        self.metrics_em  = {};    self.metrics_fcm = {}
        self.time_em     = 0.0;   self.time_fcm    = 0.0
        self.n_iter_em   = 0;     self.n_iter_fcm  = 0
        self.model_fcm   = None;  self.scaler_fcm  = None
        self._res_fig    = None;  self._cbar_ref   = [None]

        if img is not None:
            self.X, self.valid = build_feature_matrix(img)
            self.rgb_display   = compute_rgb_display(img)
        else:
            self.X = self.valid = self.rgb_display = None

        self._build_ui()

    # ── Layout helpers ────────────────────────────────────────────────────
    def _fp(self, rx, ry, rw, rh):
        return [self.CP_L + rx * self.CP_W, 0.02 + ry * 0.96,
                rw * self.CP_W, rh * 0.96]

    def _style_ax(self, ax):
        ax.set_facecolor("#161B22")
        for sp in ax.spines.values(): sp.set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=7)

    def _label(self, rx, ry, rw, text, color="#C9D1D9", fs=8.5):
        ax = self.fig.add_axes(self._fp(rx, ry, rw, 0.036))
        ax.axis("off")
        ax.text(0, 0.5, text, color=color, fontsize=fs,
                fontweight="bold", va="center")

    def _slider(self, rx, ry, rw, vmin, vmax, valinit, color, valstep=None):
        ax = self.fig.add_axes(self._fp(rx, ry, rw, 0.037))
        ax.set_facecolor("#21262D")
        kw = dict(color=color)
        if valstep is not None:
            kw["valstep"] = valstep
        sl = Slider(ax, "", vmin, vmax, valinit=valinit, **kw)
        sl.valtext.set_color(color); sl.valtext.set_fontsize(9)
        return sl

    def _hline(self, ry):
        y = 0.02 + ry * 0.96
        self.fig.add_artist(
            plt.Line2D([self.CP_L + 0.01, self.CP_R - 0.01], [y, y],
                       color="#30363D", linewidth=1,
                       transform=self.fig.transFigure))

    def _build_ui(self):
        self.fig = plt.figure(figsize=(17, 9.0), facecolor="#0D1117")
        self.fig.canvas.manager.set_window_title(
            "Satellite Image Clustering — Controls  |  GNR605")

        # ── Image panel ───────────────────────────────────────────────────
        img_gs = gridspec.GridSpec(1, 1, figure=self.fig,
                                   left=0.02, right=0.545,
                                   top=0.93, bottom=0.115)
        self.ax_orig = self.fig.add_subplot(img_gs[0, 0])
        self._style_ax(self.ax_orig)
        self._draw_image_panel()

        # ── Suptitle ──────────────────────────────────────────────────────
        self._update_suptitle()

        # ── Browse button ─────────────────────────────────────────────────
        ax_br = self.fig.add_axes([0.02, 0.025, 0.525, 0.065])
        self.btn_browse = Button(ax_br, "📂  Browse / Load Image",
                                 color="#21262D", hovercolor="#30363D")
        self.btn_browse.label.set_color("#58A6FF")
        self.btn_browse.label.set_fontsize(10)
        self.btn_browse.on_clicked(self._on_browse)
        self.fig._btn_browse = self.btn_browse

        # ── Control panel background ──────────────────────────────────────
        cp_gs = gridspec.GridSpec(1, 1, figure=self.fig,
                                  left=self.CP_L - 0.005, right=self.CP_R,
                                  top=0.97, bottom=0.02)
        self.ax_ctrl = self.fig.add_subplot(cp_gs[0, 0])
        self.ax_ctrl.set_facecolor("#161B22"); self.ax_ctrl.axis("off")
        for sp in self.ax_ctrl.spines.values(): sp.set_color("#30363D")
        self.ax_ctrl.text(0.5, 0.985, "CONTROLS",
                          transform=self.ax_ctrl.transAxes,
                          ha="center", va="top", color="#58A6FF",
                          fontsize=12, fontweight="bold", fontfamily="monospace")
        self.ax_ctrl.plot([0, 1], [0.948, 0.948], color="#30363D",
                          linewidth=1, transform=self.ax_ctrl.transAxes)

        # ════════════════════════════════════════════════════════════════
        #  SECTION 1 — Common
        # ════════════════════════════════════════════════════════════════
        self._label(0.03, 0.895, 0.94, "Algorithm", color="#58A6FF", fs=9)

        ax_radio = self.fig.add_axes(self._fp(0.03, 0.775, 0.94, 0.118))
        ax_radio.set_facecolor("#161B22")
        self.radio_algo = RadioButtons(
            ax_radio, ["EM (GMM)", "Fuzzy C-Means", "Both"],
            active=2, activecolor="#58A6FF")
        for lb in self.radio_algo.labels:
            lb.set_color("#C9D1D9"); lb.set_fontsize(9)
        self.radio_algo.on_clicked(lambda v: setattr(self, "algorithm", v))

        self._label(0.03, 0.735, 0.94, "Clusters  (k)")
        self.sl_k = self._slider(0.03, 0.688, 0.94,
                                  2, 8, DEFAULT_K, "#58A6FF", valstep=1)
        self.sl_k.on_changed(lambda v: setattr(self, "n_clusters", int(v)))

        n_v = int(np.sum(self.valid)) if self.valid is not None else 0
        self._label(0.03, 0.646, 0.94, "Sample Fraction  (%)")
        self.sl_s = self._slider(0.03, 0.599, 0.94,
                                  10, 100, int(DEFAULT_SAMPLE_FRAC * 100),
                                  "#3FB950", valstep=5)
        self._sinfo = self.fig.text(
            self.CP_L + 0.03 * self.CP_W, 0.02 + 0.576 * 0.96,
            self._npix_str(DEFAULT_SAMPLE_FRAC, n_v),
            color="#8B949E", fontsize=7, fontfamily="monospace")

        def _on_sf(v):
            frac = float(v) / 100.0
            setattr(self, "sample_frac", frac)
            n = int(np.sum(self.valid)) if self.valid is not None else 0
            self._sinfo.set_text(self._npix_str(frac, n))
            self.fig.canvas.draw_idle()
        self.sl_s.on_changed(_on_sf)

        self._hline(0.553)

        # ════════════════════════════════════════════════════════════════
        #  SECTION 2 — Algorithm-specific (2 columns)
        #  Left  (LX, LW) : GMM   |   Right (RX, LW) : FCM
        # ════════════════════════════════════════════════════════════════
        LW, LX, RX = 0.45, 0.03, 0.52

        # ── GMM ───────────────────────────────────────────────────────
        self._label(LX, 0.520, LW, "── GMM Parameters", color="#58A6FF", fs=8)
        self._label(LX, 0.483, LW, "Covariance Type", fs=7.5)

        ax_cov = self.fig.add_axes(self._fp(LX, 0.365, LW, 0.116))
        ax_cov.set_facecolor("#161B22")
        self.radio_cov = RadioButtons(
            ax_cov, ["full", "diag", "tied", "spherical"],
            active=0, activecolor="#58A6FF")
        for lb in self.radio_cov.labels:
            lb.set_color("#C9D1D9"); lb.set_fontsize(8)
        self.radio_cov.on_clicked(lambda v: setattr(self, "gmm_cov_type", v))

        self._label(LX, 0.325, LW, "n_init  (random restarts)", fs=7.5)
        self.sl_ni = self._slider(LX, 0.278, LW,
                                   1, 20, DEFAULT_GMM_N_INIT, "#58A6FF", valstep=1)
        self.sl_ni.on_changed(lambda v: setattr(self, "gmm_n_init", int(v)))

        # ── FCM ───────────────────────────────────────────────────────
        self._label(RX, 0.520, LW, "── FCM Parameters", color="#F78166", fs=8)
        self._label(RX, 0.483, LW, "Fuzziness  (m)", fs=7.5)
        self.sl_m = self._slider(RX, 0.436, LW, 1.1, 4.0, DEFAULT_FCM_M, "#F78166")
        self.sl_m.on_changed(lambda v: setattr(self, "fuzz_m", float(v)))

        self._label(RX, 0.388, LW, "Init Method", fs=7.5)
        ax_init = self.fig.add_axes(self._fp(RX, 0.278, LW, 0.108))
        ax_init.set_facecolor("#161B22")
        self.radio_init = RadioButtons(
            ax_init, ["kmeans++", "random"],
            active=0, activecolor="#F78166")
        for lb in self.radio_init.labels:
            lb.set_color("#C9D1D9"); lb.set_fontsize(8.5)
        self.radio_init.on_clicked(lambda v: setattr(self, "fcm_init", v))

        # ════════════════════════════════════════════════════════════════
        #  RUN / SAVE buttons
        # ════════════════════════════════════════════════════════════════
        ax_run = self.fig.add_axes(
            [self.CP_L, 0.025, self.CP_W * 0.57, 0.068])
        self.btn_run = Button(ax_run, "▶  RUN CLUSTERING",
                              color="#1F6FEB", hovercolor="#388BFD")
        self.btn_run.label.set_color("white")
        self.btn_run.label.set_fontsize(11)
        self.btn_run.label.set_fontweight("bold")
        self.btn_run.on_clicked(self._on_run)

        ax_sav = self.fig.add_axes(
            [self.CP_L + self.CP_W * 0.60, 0.025, self.CP_W * 0.38, 0.068])
        self.btn_save = Button(ax_sav, "💾  SAVE",
                               color="#238636", hovercolor="#2EA043")
        self.btn_save.label.set_color("white")
        self.btn_save.label.set_fontsize(10)
        self.btn_save.on_clicked(self._on_save)

        self.status = self.fig.text(
            0.28, 0.006, "Click  📂 Browse  to load a satellite image",
            ha="center", va="bottom", color="#58A6FF",
            fontsize=9, fontweight="bold", fontfamily="monospace")

    # ── Image panel helpers ───────────────────────────────────────────────
    def _draw_image_panel(self):
        self.ax_orig.cla()
        self._style_ax(self.ax_orig)
        if self.rgb_display is not None:
            self.ax_orig.imshow(self.rgb_display)
            self.ax_orig.set_title("Input Image (True Colour)",
                                   color="#C9D1D9", fontsize=9, pad=4)
        else:
            # Placeholder — dark grey canvas with instruction text
            placeholder = np.full((4, 4, 3), 0.09, dtype=np.float32)
            self.ax_orig.imshow(placeholder, aspect="auto")
            self.ax_orig.text(0.5, 0.56, "No image loaded",
                              transform=self.ax_orig.transAxes,
                              ha="center", va="center",
                              color="#58A6FF", fontsize=16, fontweight="bold")
            self.ax_orig.text(0.5, 0.44, "Click  📂 Browse  to select a file",
                              transform=self.ax_orig.transAxes,
                              ha="center", va="center",
                              color="#8B949E", fontsize=11)
            self.ax_orig.set_title("Input Image",
                                   color="#C9D1D9", fontsize=9, pad=4)
        self.ax_orig.axis("off")

    def _update_suptitle(self):
        if self.image_path and self.orig_size:
            H0, W0, B0 = self.orig_size
            title = (f"Image Clustering  ·  {os.path.basename(self.image_path)}"
                     f"  ·  {H0}×{W0}  bands={B0}")
        else:
            title = "Image Clustering  ·  GNR605 Assignment"
        self.fig.suptitle(title, color="#C9D1D9", fontsize=10,
                          fontweight="bold", y=0.98)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _npix_str(self, frac, n_valid):
        if n_valid == 0:
            return "load an image first"
        n = max(min(int(n_valid * frac), MAX_PIXELS), min(n_valid, 5000))
        return f"≈ {n:,} pixels for training"

    def _set_status(self, msg, color="#58A6FF"):
        self.status.set_text(msg); self.status.set_color(color)
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def _run_params(self):
        return dict(
            image=os.path.basename(self.image_path) if self.image_path else "—",
            algorithm=self.algorithm,
            k=self.n_clusters,
            sample_pct=f"{self.sample_frac*100:.1f}",
            gmm_cov_type=self.gmm_cov_type,
            gmm_n_init=self.gmm_n_init,
            fuzz_m=self.fuzz_m,
            fcm_init=self.fcm_init)

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _on_browse(self, _):
        new_path = select_image_file()
        if new_path is None:
            return
        try:
            self._set_status("⏳  Loading image …")
            new_img, new_orig, new_disp, new_tf, new_crs = \
                load_image(new_path, max_dim=700)
            self.img          = new_img
            self.orig_size    = new_orig
            self.display_size = new_disp
            self.H, self.W    = new_disp
            self.image_path   = new_path
            self.transform    = new_tf
            self.crs          = new_crs
            self.output_dir   = os.path.join(
                os.path.dirname(new_path), "clustering_output")
            os.makedirs(self.output_dir, exist_ok=True)
            self.X, self.valid   = build_feature_matrix(new_img)
            self.rgb_display     = compute_rgb_display(new_img)
            self.labels_em = self.labels_fcm = None
            self.metrics_em = {}; self.metrics_fcm = {}

            self._draw_image_panel()
            self._update_suptitle()
            self._sinfo.set_text(
                self._npix_str(self.sample_frac, int(np.sum(self.valid))))
            self._set_status(
                f"✓  Loaded: {os.path.basename(new_path)}", color="#3FB950")
            self.fig.canvas.draw()
        except Exception as e:
            self._set_status(f"⚠  Load error: {e}", color="#F78166")

    def _on_run(self, _):
        if self.img is None:
            self._set_status("⚠  No image loaded — click Browse first.",
                             color="#F78166")
            return

        k    = self.n_clusters
        algo = self.algorithm
        run_em  = algo in ("EM (GMM)", "Both")
        run_fcm = algo in ("Fuzzy C-Means", "Both")

        if run_em:
            self._set_status("⏳  [1/2] Fitting EM/GMM …")
            lbl, mdl, scl, t, ni = run_clustering(
                self.X, self.valid, self.H, self.W, k, "EM (GMM)",
                gmm_cov_type=self.gmm_cov_type,
                gmm_n_init=self.gmm_n_init,
                sample_frac=self.sample_frac)
            self.labels_em  = lbl;  self.time_em  = t;  self.n_iter_em  = ni
            self.metrics_em = compute_metrics(
                self.X, self.valid, lbl, "EM (GMM)", mdl, scl)
        else:
            self.labels_em = None; self.metrics_em = {}
            self.time_em = 0.0;    self.n_iter_em  = 0

        if run_fcm:
            step = "2/2" if run_em else "1/1"
            self._set_status(f"⏳  [{step}] Fitting FCM …")
            lbl, mdl, scl, t, ni = run_clustering(
                self.X, self.valid, self.H, self.W, k, "Fuzzy C-Means",
                fuzz_m=self.fuzz_m,
                fcm_init=self.fcm_init,
                sample_frac=self.sample_frac)
            self.labels_fcm  = lbl;  self.time_fcm  = t;  self.n_iter_fcm  = ni
            self.model_fcm   = mdl;  self.scaler_fcm = scl
            self.metrics_fcm = compute_metrics(
                self.X, self.valid, lbl, "Fuzzy C-Means", mdl, scl)
        else:
            self.labels_fcm = None; self.metrics_fcm = {}
            self.time_fcm = 0.0;    self.n_iter_fcm  = 0
            self.model_fcm = None;  self.scaler_fcm  = None

        if self.labels_em is not None and self.labels_fcm is not None:
            self._set_status("⏳  Aligning cluster labels …")
            self.labels_fcm = align_labels(self.labels_em, self.labels_fcm, k)

        self._set_status("⏳  Opening results window …")
        if self._res_fig is not None:
            try: plt.close(self._res_fig)
            except Exception: pass

        self._res_fig = open_results_window(
            self.rgb_display,
            self.labels_em, self.labels_fcm, k,
            self.time_em, self.time_fcm,
            self.n_iter_em, self.n_iter_fcm,
            self.metrics_em, self.metrics_fcm,
            self.img, self._cbar_ref,
            output_dir=self.output_dir,
            run_params=self._run_params(),
            fcm_model=self.model_fcm,
            fcm_scaler=self.scaler_fcm,
            X=self.X, valid=self.valid)

        self._set_status(
            f"✓  Done  k={k}  algo={algo}  sample={self.sample_frac*100:.0f}%",
            color="#3FB950")

    def _on_save(self, _):
        if self.labels_em is None and self.labels_fcm is None:
            self._set_status("⚠  Run clustering first.", color="#F78166")
            return
        import tkinter as tk
        from tkinter import filedialog
        import datetime
        rt = tk.Tk(); rt.withdraw(); rt.attributes("-topmost", True)
        d = filedialog.askdirectory(
            title="Select folder to save results",
            initialdir=self.output_dir or os.path.expanduser("~"))
        rt.destroy()
        if not d:
            self._set_status("⚠  Save cancelled.", color="#F78166"); return

        ts_c = datetime.datetime.now().strftime("%d%m_%H%M")
        cp = os.path.join(d, f"control_{ts_c}.png")
        self.fig.savefig(cp, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        print(f"\n  ✓ Control PNG        → {cp}")

        ts = save_all_results(
            d, self.labels_em, self.labels_fcm, self.n_clusters,
            self.time_em, self.time_fcm, self.n_iter_em, self.n_iter_fcm,
            self.metrics_em, self.metrics_fcm,
            self.X, self.valid,
            self.rgb_display,
            cluster_names=None, run_params=self._run_params(),
            fig_results=self._res_fig)

        self._set_status(f"✓  Saved [{ts}] → {os.path.basename(d)}/",
                         color="#3FB950")
        self.fig.canvas.draw()

    def show(self):
        plt.show()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN — launches directly, no startup dialog                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    print("=" * 65)
    print("  Satellite Image Clustering — EM (GMM) vs Fuzzy C-Means  v5")
    print("  GNR605 Assignment")
    print("=" * 65)
    print("\n  Launching dashboard … (use Browse button to load an image)")

    app = ClusteringApp()   # no image — shows placeholder
    plt.show()
    print("\n✓ Session complete.")


if __name__ == "__main__":
    main()