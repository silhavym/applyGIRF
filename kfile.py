"""
Update a rapid-MRSI k-space file (CSV: TR,Kx,Ky,time) with the GIRF-predicted trajectory.

Only the Kx and Ky columns are rewritten. TR, time and any other column are
copied through exactly as they were in the original file.
Kx, Ky are in mm^-1 (cycles/mm); the GIRF code works in cycles/m, hence the /1e3.
"""
import csv
import os
import numpy as np
from io_helper import read_dsv


def read_kfile(path):
    """Return (header, rows-as-strings, column-index dict keyed by lower-case name)."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    header = [h.strip() for h in rows[0]]
    col = {h.lower(): i for i, h in enumerate(header)}
    missing = [c for c in ("tr", "kx", "ky", "time") if c not in col]
    if missing:
        raise ValueError(f"{path}: expected columns TR,Kx,Ky,time but found {header}")
    return header, rows[1:], col


def adc_start_from_dsv(path):
    """Time (s) of the first ADC-active sample in a Siemens *_ADC.dsv file."""
    a, dt, _ = read_dsv(path)
    active = np.flatnonzero(np.abs(a) > 0.5 * np.abs(a).max())
    if len(active) == 0:
        raise ValueError(f"No ADC activity found in {path}")
    return active[0] * dt


def _k_nodes(out, axis, kind):
    """k(t) as piecewise-linear nodes: k = cumsum(g)*dt is the value at the END of each sample."""
    dt = float(out[f"dt_{axis}"])
    t = np.concatenate([[0.0], out[f"t_{axis}"] + dt])
    k = np.concatenate([[0.0], out[f"k{axis}_{kind}"]])
    return t, k


def update_kfile(kfile_in, kfile_out, out, adc_start_s, time_scale=1.0, mode="replace"):
    """
    out         : dict built by apply.py (needs the x and y axes)
    adc_start_s : time of the first ADC sample, relative to the start of the DSV (s)
    time_scale  : multiply the k-file 'time' column by this to get seconds
    mode        : 'replace' -> Kx,Ky = GIRF-predicted k at the ADC times
                  'delta'   -> Kx,Ky = old Kx,Ky + (GIRF k - nominal DSV k)
    """
    header, rows, col = read_kfile(kfile_in)
    t_q = adc_start_s + time_scale * np.array([float(r[col["time"]]) for r in rows])

    for a, name in (("x", "kx"), ("y", "ky")):
        tk, k_nom = _k_nodes(out, a, "nom")
        _, k_act = _k_nodes(out, a, "act")
        if t_q.min() < 0 or t_q.max() > tk[-1]:
            raise ValueError(
                f"k-file sample times span {t_q.min()*1e3:.3f}..{t_q.max()*1e3:.3f} ms but the "
                f"DSV waveform spans 0..{tk[-1]*1e3:.3f} ms. Check --adc_start_us and "
                f"--k_time_scale (use 1e-3 if the 'time' column is in ms, 1e-6 if in us).")
        nom = np.interp(t_q, tk, k_nom) / 1e3
        act = np.interp(t_q, tk, k_act) / 1e3
        old = np.array([float(r[col[name]]) for r in rows])
        new = act if mode == "replace" else old + (act - nom)

        rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
        print(f"[K{a}] file vs DSV-nominal: rms diff {rms(nom - old):.4g} mm^-1 "
              f"(file rms {rms(old):.4g});  GIRF shift rms {rms(act - nom):.4g} mm^-1")
        if rms(nom - old) > 0.1 * rms(old):
            print(f"    WARNING: the DSV nominal trajectory does not match the k-file for K{a}. "
                  f"Check --adc_start_us/--dsv_adc, --k_time_scale, and the sign/axis mapping, "
                  f"or use --k_mode delta.")
        for r, v in zip(rows, new):
            r[col[name]] = f"{v:.10g}"

    with open(kfile_out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved {kfile_out}  ({len(rows)} rows, mode={mode})")


def default_out_name(kfile_in):
    base, ext = os.path.splitext(kfile_in)
    return f"{base}_girf{ext or '.txt'}"
