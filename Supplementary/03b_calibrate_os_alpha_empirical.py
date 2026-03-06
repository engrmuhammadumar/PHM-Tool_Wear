# %% [markdown]
# # 03b — Empirical OS-CFAR alpha calibrator
# Finds calm seconds (S4–S7), computes r = X / Qq(training) on those,
# and writes alpha to os_cfar_alpha_cache.json for your notebooks to use.

# %%
import os, json, numpy as np
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view

# --- Config (match your earlier notebooks) ---
DATA_DIR = r"D:\Pipeline RUL Data"
FILE = "B.wfs"          # calibrate on one pipe (B) first
N_CHANNELS = 8
DTYPE = np.dtype("<i2")
HEADER_BYTES = 2
FS = 1_000_000
CELL = 500

SENSORS_FOR_CALM = [4,5,6,7]   # use these to select calm seconds
CALIBRATE_SENSOR = 4           # compute alpha from this sensor's calm segments

# CFAR knobs (must match your use later)
T = 50
G = 4
Q = 0.70
TARGET_PFA = 1e-4

# Search window for calm detection
SCAN_MINUTES = 60        # scan first 60 minutes (increase if you like)
TOP_K_SECONDS = 120      # use this many calm seconds for calibration

ALPHA_CACHE_PATH = "os_cfar_alpha_cache.json"

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

# %%
def find_calm_seconds(path, sensors, scan_minutes, top_k):
    cells_per_sec = FS // CELL
    need_cells = int(scan_minutes*60 * cells_per_sec)
    buckets = {s: defaultdict(list) for s in sensors}
    seen = 0
    for start, block in iter_cell_power(path, seconds_per_chunk=10.0):
        n = block.shape[1]
        if seen >= need_cells: break
        if seen + n > need_cells:
            n = need_cells - seen
            block = block[:, :n]
        secs = (start + np.arange(n)) // cells_per_sec
        uniq = np.unique(secs)
        for s in sensors:
            x = block[s]
            for u in uniq:
                m = (secs == u)
                buckets[s][int(u)].append(x[m])
        seen += n

    seconds = sorted(set().union(*[set(buckets[s].keys()) for s in sensors]))
    scored = []
    for sec in seconds:
        spreads = []
        ok = True
        for s in sensors:
            if sec not in buckets[s]:
                ok=False; break
            arr = np.concatenate(buckets[s][sec])
            if arr.size < 100: ok=False; break
            q1, q3 = np.percentile(arr, [25, 75])
            spreads.append(float(q3 - q1))
        if ok:
            scored.append((max(spreads), sec))
    scored.sort(key=lambda t: t[0])
    return [sec for _, sec in scored[:top_k]], buckets

def ratios_for_segments(segments, L, T, G, q):
    """Compute r = X / Qq(training) for centers within each segment."""
    W = T + G; K = 2*T
    qi = int(np.floor(q*(K-1)))
    rs = []
    for seg in segments:
        x = np.asarray(seg, float)
        if x.size < 2*W+1: continue
        win = sliding_window_view(x, 2*W+1)           # (N-2W, 2W+1)
        idx = np.arange(2*W+1); mask = (np.abs(idx - W) > G)
        train = win[:, mask]                          # (N-2W, K)
        part = np.partition(train, qi, axis=1)
        qv = part[:, qi]
        X = x[W:len(x)-W]
        r = X / qv
        rs.append(r)
    return np.concatenate(rs) if rs else np.empty((0,), float)

# %%
path = os.path.join(DATA_DIR, FILE)
calm_secs, buckets = find_calm_seconds(path, SENSORS_FOR_CALM, SCAN_MINUTES, TOP_K_SECONDS)
if not calm_secs:
    raise SystemExit("No calm seconds found. Increase SCAN_MINUTES.")

print(f"Using {len(calm_secs)} calm seconds (first few): {calm_secs[:10]}")
segs = [np.concatenate(buckets[CALIBRATE_SENSOR][sec]) for sec in calm_secs]
r = ratios_for_segments(segs, L=CELL, T=T, G=G, q=Q)
if r.size == 0:
    raise SystemExit("No valid centers in selected segments.")

alpha_emp = float(np.quantile(r, 1.0 - TARGET_PFA))
print(f"Empirical OS-CFAR alpha for PFA={TARGET_PFA:g} (L={CELL}, T={T}, G={G}, q={Q}): {alpha_emp:.6f}")

key = f"L={CELL}|K={2*T}|q={Q}|pfa={TARGET_PFA}"
cache = {}
if os.path.exists(ALPHA_CACHE_PATH):
    try:
        with open(ALPHA_CACHE_PATH, "r") as f: cache = json.load(f)
    except Exception:
        cache = {}
cache[key] = alpha_emp
tmp = ALPHA_CACHE_PATH + ".tmp"
with open(tmp, "w") as f: json.dump(cache, f, indent=2)
os.replace(tmp, ALPHA_CACHE_PATH)
print(f"Wrote {ALPHA_CACHE_PATH} with key '{key}'. Re-run 03/04 to use cached alpha.")
