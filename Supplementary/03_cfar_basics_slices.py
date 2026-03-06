# %% [markdown]
# # 03 — CFAR Basics (CA & OS) on Slices
#
# - Implements CA-CFAR & OS-CFAR on 1-D cell-power
# - Uses calm seconds from Notebook 02
# - Monte-Carlo alpha and optional cached alpha (from a separate calibrator if you make one)

# %%
import os, json, numpy as np, matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict

# Paths & constants
DATA_DIR = r"D:\Pipeline RUL Data"
FILE = "B.wfs"
N_CHANNELS = 8
DTYPE = np.dtype("<i2"); HEADER_BYTES = 2
FS = 1_000_000; CELL = 500
SENSORS = [4,5,6,7]

# CFAR params
T = 50; G = 4; PFA = 1e-4; Q = 0.70
USE_CACHED_OS_ALPHA = True
ALPHA_CACHE = "os_cfar_alpha_cache.json"   # optional cache

# %%
def iter_cell_power(path, seconds_per_chunk=10.0):
    frame_bytes = N_CHANNELS * DTYPE.itemsize
    cell_bytes = frame_bytes * CELL
    n_cells = int(seconds_per_chunk * FS // CELL)
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES, os.SEEK_SET)
        start_cell = 0
        while True:
            raw = f.read(n_cells * cell_bytes)
            if not raw: break
            arr = np.frombuffer(raw, dtype=DTYPE)
            if arr.size == 0: break
            cells = (arr.size // (N_CHANNELS*CELL))
            if cells == 0: break
            arr = arr[:cells*N_CHANNELS*CELL].reshape(cells, N_CHANNELS, CELL)
            cp = (arr.astype(np.float32)**2).mean(axis=2)  # [cells, ch]
            yield start_cell, cp.T.copy()
            start_cell += cells

def find_calm_seconds(path, sensors, minutes=10):
    cells_per_sec = FS // CELL
    need_cells = int(minutes*60 * cells_per_sec)
    buckets = {s: defaultdict(list) for s in sensors}
    seen=0
    for start, block in iter_cell_power(path, seconds_per_chunk=10.0):
        n = block.shape[1]
        if seen >= need_cells: break
        if seen + n > need_cells:
            n = need_cells - seen; block = block[:, :n]
        secs = (start + np.arange(n)) // cells_per_sec
        uniq = np.unique(secs)
        for s in sensors:
            x = block[s]
            for u in uniq:
                m = (secs == u)
                buckets[s][int(u)].append(x[m])
        seen += n
    seconds = sorted(set().union(*[set(buckets[s].keys()) for s in sensors]))
    scores = []
    for sec in seconds:
        spreads = []; ok = True
        for s in sensors:
            if sec not in buckets[s]:
                ok=False; break
            arr = np.concatenate(buckets[s][sec])
            if arr.size < 100: ok=False; break
            q1, q3 = np.percentile(arr, [25, 75])
            spreads.append(float(q3-q1))
        if ok: scores.append((max(spreads), sec))
    scores.sort(key=lambda t: t[0])
    return [sec for _,sec in scores], buckets

def build_series_from_seconds(buckets, sensor, seconds):
    cells_per_sec = FS // CELL
    cp = []
    for sec in seconds:
        seg = np.concatenate(buckets[sensor][sec])
        cp.append(seg)
    x = np.concatenate(cp) if cp else np.array([])
    return x

# %%
def alpha_ca_mc(pfa, L, K, N=300_000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L*K, size=N)/(L*K)
    R = X / Y
    return float(np.quantile(R, 1.0 - pfa))

def alpha_os_mc(pfa, L, K, q, N=300_000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L, size=(N, K))/L
    qv = np.quantile(Y, q, axis=1)
    R = X / qv
    return float(np.quantile(R, 1.0 - pfa))

def cfar_ca_1d(x, L, T, G, alpha):
    W = T + G
    k = np.arange(-W, W+1)
    ker = ((np.abs(k) > G) & (np.abs(k) <= G+T)).astype(float)
    s = np.convolve(x, ker, mode="same"); c = np.convolve(np.ones_like(x), ker, mode="same")
    c = np.maximum(c, 1.0); thr = alpha * (s/c)
    valid = (c >= 0.8*(2*T))
    det = (x > thr) & valid
    return thr, det, valid

def cfar_os_1d(x, L, T, G, alpha, q):
    W = T + G; K = 2*T
    if x.size < 2*W+1:
        thr = np.full_like(x, np.nan, float)
        det = np.zeros_like(x, bool)
        return thr, det, np.isfinite(thr)
    win = sliding_window_view(x, 2*W+1)         # (N-2W, 2W+1)
    idx = np.arange(2*W+1); mask = (np.abs(idx - W) > G)
    train = win[:, mask]                        # (N-2W, K)
    qi = int(np.floor(q*(K-1)))
    part = np.partition(train, qi, axis=1)
    qv = part[:, qi]
    centers = np.arange(W, len(x)-W)
    thr = np.full_like(x, np.nan, float)
    thr_core = alpha * qv
    thr[centers] = thr_core
    det = np.zeros_like(x, bool)
    det[centers] = x[centers] > thr_core
    valid = np.isfinite(thr)
    return thr, det, valid

# %%
path = os.path.join(DATA_DIR, FILE)
calm_secs, buckets = find_calm_seconds(path, SENSORS, minutes=10)
calm_top = calm_secs[:30] if calm_secs else []
print("Picked calm seconds:", calm_top[:10], "... total", len(calm_top))

x = build_series_from_seconds(buckets, sensor=4, seconds=calm_top)
print("Series length:", x.size, "cells")

L = CELL ; K = 2*T
a_ca = alpha_ca_mc(PFA, L, K)

# OS alpha: read cache if present, otherwise MC
a_os = None
if USE_CACHED_OS_ALPHA:
    key = f"L={L}|K={K}|q={Q}|pfa={PFA}"
    try:
        with open(ALPHA_CACHE, "r") as f:
            cache = json.load(f)
        if key in cache:
            a_os = float(cache[key])
            print(f"Using cached OS alpha for {key}: {a_os:.6f}")
    except Exception:
        pass
if a_os is None:
    a_os = alpha_os_mc(PFA, L, K, Q)
    print(f"No cached alpha; Monte-Carlo OS alpha≈{a_os:.6f}")

if x.size > 2*(T+G)+10:
    thr_ca, det_ca, val_ca = cfar_ca_1d(x, L, T, G, a_ca)
    thr_os, det_os, val_os = cfar_os_1d(x, L, T, G, a_os, Q)
    p_ca = det_ca[val_ca].mean() if val_ca.any() else np.nan
    p_os = det_os[val_os].mean() if val_os.any() else np.nan
    print(f"Empirical hit fraction on calm slice (target PFA={PFA:g}): CA={p_ca:.3e}, OS={p_os:.3e}")

    # Plot small excerpt
    N = min(6000, x.size)
    t = np.arange(N)/(FS//CELL)
    fig, ax = plt.subplots(1,1, figsize=(12,3))
    ax.plot(t, x[:N], lw=0.8, label="cell power")
    ax.plot(t, thr_os[:N], lw=0.8, label="OS thr")
    ax.plot(t, thr_ca[:N], lw=0.8, label="CA thr")
    ax.set_xlabel("time (s)"); ax.set_ylabel("counts²")
    ax.legend(); plt.tight_layout(); plt.show()
else:
    print("Not enough calm cells; increase minutes or TOP-K.")
