"""
2D k-file generation for nominal ("nom") and GIRF-predicted ("girf") waveforms.

Format (one row per ADC sample):
    tr Kx Ky
where `tr` is the 1-based rosette leaf index.
"""
import os
import numpy as np


def resolve_kfile_paths(base_kfile):
    """Return (nom_path, girf_path) from --kfile."""
    root, ext = os.path.splitext(base_kfile)
    ext = ext or ".kfile"
    return f"{root}_nom{ext}", f"{root}_girf{ext}"


def adc_trajectory(out, kind, t_adc_s, n_adc=None, start_us=0.0):
    """
    Sample 2D k-space (Kx, Ky) at ADC times `t_adc_s` (in seconds).

    kind : 'nom' or 'girf' (maps to 'nom' or 'act' arrays in `out`)
    """
    data_kind = "act" if kind == "girf" else "nom"
    axes = ("x", "y")

    ks, t_end = [], np.inf
    for a in axes:
        if f"t_{a}" not in out or f"k{a}_{data_kind}" not in out:
            raise ValueError(f"Missing axis '{a}' ({data_kind}) required for 2D Kx Ky k-file")
        t = np.atleast_1d(out[f"t_{a}"]).astype(float)
        dt = float(out[f"dt_{a}"])
        tk = np.concatenate([[0.0], t + dt])
        kk = np.concatenate([[0.0], np.atleast_1d(out[f"k{a}_{data_kind}"]).astype(float)])
        ks.append((tk, kk))
        t_end = min(t_end, tk[-1])

    t_adc = np.asarray(t_adc_s, dtype=float) + start_us * 1e-6
    if n_adc is not None:
        t_adc = t_adc[:n_adc]

    if len(t_adc) > 0 and t_adc[-1] > t_end + 1e-12:
        raise ValueError(f"ADC end time {t_adc[-1]:.6f} s exceeds gradient end time {t_end:.6f} s")

    k = np.stack([np.interp(t_adc, tk, kk) for tk, kk in ks], axis=1)
    return k, t_adc


def build_kfile(
    out,
    kind,
    t_adc_s,
    n_adc=None,
    start_us=0.0,
    points_per_leaf=176,
    n_leaves=None,
):
    """
    Build k-file data with columns `tr Kx Ky` (1-based `tr`).
    """
    k, t_adc = adc_trajectory(out, kind=kind, t_adc_s=t_adc_s, n_adc=n_adc, start_us=start_us)
    n = len(t_adc)

    if points_per_leaf is not None and points_per_leaf > 0:
        tr = np.floor(np.arange(n) / points_per_leaf).astype(int) + 1
        if n_leaves is not None and n_leaves > 0:
            tr = np.minimum(tr, n_leaves)
    elif n_leaves is not None and n_leaves > 0:
        tr = np.floor(np.arange(n) * n_leaves / max(n, 1)).astype(int) + 1
        tr = np.minimum(tr, n_leaves)
    else:
        tr = np.ones(n, dtype=int)

    return {
        "kind": kind,
        "t_adc": t_adc,
        "tr": tr,
        "k": k,
    }


def write_kfile(path, kf):
    """Write `tr Kx Ky` rows to `path`."""
    tr = np.atleast_1d(kf["tr"]).astype(int)
    k = np.asarray(kf["k"], dtype=float)
    data = np.column_stack([tr, k[:, 0], k[:, 1]])
    np.savetxt(path, data, fmt=["%d", "%.8e", "%.8e"])