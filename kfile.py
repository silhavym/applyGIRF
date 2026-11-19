"""
k-file: k-space location and gradient value at every ADC sample,
for both the nominal and the GIRF-predicted waveforms.

Text format (one row per ADC sample, header lines start with '#'):

    # GIRF k-file v1
    # n_points 2000
    # adc_dwell_us 10.0
    # adc_start_us 0.0
    # k_units cycles/m
    # g_units mT/m
    # columns idx t_us kx_nom ky_nom kx_act ky_act gx_nom gy_nom gx_act gy_act
    0 0.0 ...

"nom" = nominal (from the DSV), "act" = GIRF-predicted.
"""
import numpy as np


def adc_times(dwell_us, start_us, n_adc):
    """ADC sample times in seconds, relative to the start of the DSV waveform."""
    return start_us * 1e-6 + np.arange(n_adc) * dwell_us * 1e-6


def _end_time(out, axes):
    return min(float(np.atleast_1d(out[f"t_{a}"])[-1]) + float(out[f"dt_{a}"]) for a in axes)


def adc_trajectory(out, axes, kind, dwell_us, start_us, n_adc=None):
    """
    Sample k(t) at the ADC times.

    kind     : 'nom' or 'act' (nominal or GIRF-predicted)
    dwell_us : ADC dwell time (us)
    start_us : time of the first ADC sample, relative to DSV start (us)
    n_adc    : number of ADC samples (default: as many as fit in the waveform)

    Returns k (n_adc, len(axes)) in cycles/m, and t_adc (s).
    """
    ks, t_end = [], np.inf
    for a in axes:
        t = np.atleast_1d(out[f"t_{a}"]).astype(float)
        dt = float(out[f"dt_{a}"])
        # k = cumsum(g)*dt is the value at the END of each gradient sample
        tk = np.concatenate([[0.0], t + dt])
        kk = np.concatenate([[0.0], np.atleast_1d(out[f"k{a}_{kind}"]).astype(float)])
        ks.append((tk, kk))
        t_end = min(t_end, tk[-1])

    if n_adc is None:
        n_adc = int((t_end - start_us * 1e-6) / (dwell_us * 1e-6) + 1e-9)
    t_adc = adc_times(dwell_us, start_us, n_adc)
    if t_adc[-1] > t_end + 1e-12:
        raise ValueError(f"ADC end time {t_adc[-1]:.6f} s exceeds gradient end time {t_end:.6f} s")
    k = np.stack([np.interp(t_adc, tk, kk) for tk, kk in ks], axis=1)
    return k, t_adc


def adc_gradient(out, axes, kind, t_adc):
    """Gradient (mT/m) in effect at each ADC time (piecewise constant, matching the k integration)."""
    gs = []
    for a in axes:
        g = np.atleast_1d(out[f"g{a}_{kind}"]).astype(float)
        dt = float(out[f"dt_{a}"])
        i = np.clip(np.floor(t_adc / dt + 1e-6).astype(int), 0, len(g) - 1)
        gs.append(g[i])
    return np.stack(gs, axis=1)


def build_kfile(out, axes, dwell_us, start_us=0.0, n_adc=None):
    """
    Build the k-file contents from the dict produced by apply.py.

    Returns dict with: axes, t_adc (s), k_nom, k_act (cycles/m), g_nom, g_act (mT/m);
    each k/g array has shape (n_adc, len(axes)).
    """
    kf = {"axes": "".join(axes)}
    for kind in ("nom", "act"):
        k, t_adc = adc_trajectory(out, axes, kind, dwell_us, start_us, n_adc)
        n_adc = len(t_adc)                      # fix n_adc after the first pass
        kf[f"k_{kind}"] = k
        kf[f"g_{kind}"] = adc_gradient(out, axes, kind, t_adc)
    kf["t_adc"] = t_adc
    return kf


def write_kfile(path, kf, dwell_us, start_us=0.0):
    axes = kf["axes"]
    n = len(kf["t_adc"])
    names = ["idx", "t_us"]
    cols = [np.arange(n), kf["t_adc"] * 1e6]
    for prefix, key in (("k", "k"), ("g", "g")):
        for kind in ("nom", "act"):
            for i, a in enumerate(axes):
                names.append(f"{prefix}{a}_{kind}")
                cols.append(kf[f"{key}_{kind}"][:, i])
    header = "\n".join([
        "GIRF k-file v1",
        f"n_points {n}",
        f"adc_dwell_us {dwell_us}",
        f"adc_start_us {start_us}",
        "k_units cycles/m",
        "g_units mT/m",
        "columns " + " ".join(names),
    ])
    fmt = ["%d"] + ["%.6f"] + ["%.8e"] * (len(cols) - 2)
    np.savetxt(path, np.column_stack(cols), fmt=fmt, header=header, comments="# ")


def read_kfile(path):
    """Read a k-file written by write_kfile. Returns the same dict layout as build_kfile, plus 'meta'."""
    meta, names = {}, None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            s = line[1:].strip()
            if s.startswith("columns "):
                names = s.split()[1:]
            elif " " in s and not s.startswith("GIRF"):
                k, v = s.split(None, 1)
                meta[k] = v
    if names is None:
        raise ValueError(f"{path}: no 'columns' header line found")
    data = np.loadtxt(path, ndmin=2)
    col = {n: data[:, i] for i, n in enumerate(names)}
    axes = "".join(a for a in "xyz" if f"k{a}_act" in col)
    kf = {"axes": axes, "t_adc": col["t_us"] * 1e-6, "meta": meta}
    for kind in ("nom", "act"):
        kf[f"k_{kind}"] = np.stack([col[f"k{a}_{kind}"] for a in axes], axis=1)
        kf[f"g_{kind}"] = np.stack([col[f"g{a}_{kind}"] for a in axes], axis=1)
    return kf
