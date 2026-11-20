"""
Generate rapid-MRSI trajectory files from nominal and GIRF-predicted gradients.

CSV columns: TR,Kx,Ky,time. Kx/Ky are cycles/mm and time is relative
to the first ADC sample. By default k=0 at the first ADC sample, matching
the convention used by FID-A's rosette tools.

Important: n_tr repeats the supplied one-pass trajectory for each TR. It
does not synthesize rotated rosette leaves. For acquisitions with a distinct
gradient waveform per leaf, supply/process each leaf separately and preserve
its TR/sample mapping.
"""
import csv
import numpy as np
from io_helper import read_dsv

TIME_UNIT = {"s": 1.0, "ms": 1e3, "us": 1e6}


def adc_window_from_dsv(path):
    """Return (start, duration), in seconds, for ADC activity in a Siemens DSV."""
    a, dt, _ = read_dsv(path)
    active = np.flatnonzero(np.abs(a) > 0.5 * np.abs(a).max())
    if len(active) == 0:
        raise ValueError(f"No ADC activity found in {path}")
    return active[0] * dt, (active[-1] - active[0] + 1) * dt


def _k_nodes(out, axis, kind):
    """Return piecewise-linear k(t) nodes; integrated samples are end-of-raster values."""
    if kind not in ("nom", "act"):
        raise ValueError("kind must be 'nom' or 'act'")
    dt = float(out[f"dt_{axis}"])
    t = np.concatenate([[0.0], out[f"t_{axis}"] + dt])
    k = np.concatenate([[0.0], out[f"k{axis}_{kind}"]])
    return t, k


def build_kfile_data(out, adc_start_s, dwell_s, n_adc, n_tr=1,
                     time_unit="s", zero_at_adc_start=True, kind="act"):
    """
    Build trajectory columns without writing a file.

    The same supplied one-pass trajectory is repeated for TR=1..n_tr.
    Returns a dict containing TR, Kx, Ky and time arrays.
    """
    if n_adc <= 0 or n_tr <= 0:
        raise ValueError("n_adc and n_tr must be positive")
    if dwell_s <= 0:
        raise ValueError("dwell_s must be positive")
    if time_unit not in TIME_UNIT:
        raise ValueError(f"Unsupported time unit: {time_unit}")

    t_rel = np.arange(n_adc, dtype=float) * dwell_s
    t_q = adc_start_s + t_rel
    cols = {}
    for axis, name in (("x", "Kx"), ("y", "Ky")):
        tk, kvals = _k_nodes(out, axis, kind)
        if t_q[0] < tk[0] or t_q[-1] > tk[-1]:
            raise ValueError(
                f"ADC window ({t_q[0]*1e3:.3f} to {t_q[-1]*1e3:.3f} ms) "
                f"falls outside {axis.upper()} gradient support "
                f"(0 to {tk[-1]*1e3:.3f} ms). Check ADC timing, dwell and sample count."
            )
        k = np.interp(t_q, tk, kvals) / 1e3  # cycles/m -> cycles/mm
        if zero_at_adc_start:
            k = k - k[0]
        cols[name] = k

    # Preserve one row per acquired sample, in TR-major/sample-minor order.
    return {
        "TR": np.repeat(np.arange(1, n_tr + 1, dtype=int), n_adc),
        "Kx": np.tile(cols["Kx"], n_tr),
        "Ky": np.tile(cols["Ky"], n_tr),
        "time": np.tile(t_rel * TIME_UNIT[time_unit], n_tr),
        "n_adc": n_adc,
        "n_tr": n_tr,
        "kind": kind,
        "time_unit": time_unit,
        "dwell_s": dwell_s,
        "adc_start_s": adc_start_s,
    }


def write_kfile(path, data):
    """Write build_kfile_data output as a four-column CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["TR", "Kx", "Ky", "time"])
        writer.writerows(zip(data["TR"], data["Kx"], data["Ky"], data["time"]))
    k_radius = np.hypot(data["Kx"], data["Ky"])
    print(
        f"Saved {path} ({data['n_adc']} points x {data['n_tr']} TR, "
        f"dwell {data['dwell_s']*1e6:.3f} us, "
        f"ADC start {data['adc_start_s']*1e6:.3f} us, "
        f"k={'zeroed at first ADC point' if np.isclose(data['Kx'][0], 0) and np.isclose(data['Ky'][0], 0) else 'absolute/relative'})"
    )
    print(f"    |K| max = {k_radius.max():.6g} cycles/mm")


def make_kfile(path, out, adc_start_s, dwell_s, n_adc, n_tr=1,
               time_unit="s", zero_at_adc_start=True, kind="act"):
    """Build and write one nominal ('nom') or GIRF ('act') k-file."""
    data = build_kfile_data(
        out, adc_start_s, dwell_s, n_adc, n_tr, time_unit,
        zero_at_adc_start, kind
    )
    write_kfile(path, data)
    return data
