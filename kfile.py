"""
Generate a rapid-MRSI k-space file (CSV: TR,Kx,Ky,time) from the GIRF-predicted trajectory.

Kx, Ky are in mm^-1 (cycles/mm); the GIRF code works in cycles/m, hence the /1e3.
'time' is the sampling time relative to the start of the ADC.
By default k is referenced so that k = 0 at the first ADC sample, which is what
FID-A's rosette tools assume for the k-file.
"""
import csv
import numpy as np
from io_helper import read_dsv

TIME_UNIT = {"s": 1.0, "ms": 1e3, "us": 1e6}      # seconds -> unit


def adc_window_from_dsv(path):
    """(start, duration) in seconds of the ADC-active period in a Siemens *_ADC.dsv file."""
    a, dt, _ = read_dsv(path)
    active = np.flatnonzero(np.abs(a) > 0.5 * np.abs(a).max())
    if len(active) == 0:
        raise ValueError(f"No ADC activity found in {path}")
    return active[0] * dt, (active[-1] - active[0] + 1) * dt


def _k_nodes(out, axis, kind):
    """k(t) as piecewise-linear nodes: k = cumsum(g)*dt is the value at the END of each sample."""
    dt = float(out[f"dt_{axis}"])
    t = np.concatenate([[0.0], out[f"t_{axis}"] + dt])
    k = np.concatenate([[0.0], out[f"k{axis}_{kind}"]])
    return t, k


def make_kfile(path, out, adc_start_s, dwell_s, n_adc, n_tr=1, time_unit="s",
               zero_at_adc_start=True):
    """
    out         : dict built by apply.py (needs the x and y axes)
    adc_start_s : time of the first ADC sample, relative to the start of the DSV (s)
    dwell_s     : ADC dwell time (s)
    n_adc       : number of ADC samples in one pass
    n_tr        : the one-pass trajectory is written this many times (TR = 1..n_tr)
    """
    t_rel = np.arange(n_adc) * dwell_s            # time relative to ADC start
    t_q = adc_start_s + t_rel
    cols = {}
    for a, name in (("x", "Kx"), ("y", "Ky")):
        tk, k_act = _k_nodes(out, a, "act")
        if t_q.max() > tk[-1]:
            raise ValueError(f"ADC ends at {t_q.max()*1e3:.3f} ms but the DSV waveform (plus "
                             f"padding) ends at {tk[-1]*1e3:.3f} ms. Check --adc_start_us, "
                             f"--dwell_us and --n_adc.")
        k = np.interp(t_q, tk, k_act) / 1e3                       # cycles/m -> mm^-1
        if zero_at_adc_start:
            k = k - k[0]
        cols[name] = k

    tcol = t_rel * TIME_UNIT[time_unit]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["TR", "Kx", "Ky", "time"])
        for tr in range(1, n_tr + 1):
            for i in range(n_adc):
                w.writerow([tr, f"{cols['Kx'][i]:.10g}", f"{cols['Ky'][i]:.10g}", f"{tcol[i]:.10g}"])
    print(f"Saved {path}  ({n_adc} points x {n_tr} TR, dwell {dwell_s*1e6:.2f} us, "
          f"ADC start {adc_start_s*1e6:.1f} us, time in {time_unit}, "
          f"k={'0 at first ADC point' if zero_at_adc_start else 'absolute'})")
    print(f"    |K| max = {np.hypot(cols['Kx'], cols['Ky']).max():.4g} mm^-1")
