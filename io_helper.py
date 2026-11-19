"""
Input/output helpers: Siemens DSV reader, GIRF loader, MATLAB .mat writer
"""

import os
import numpy as np
from scipy.io import savemat


def _dsv_decompress(raw):
    """
    Undo the Siemens DSV compression (same scheme as FID-A's op_dsv2timecourse.m).

    The numbers stored under [VALUES] are sample-to-sample differences.
    Two equal numbers in a row are followed by a count of further repeats.
    A cumulative sum of the expanded stream gives the waveform.
    """
    raw = np.asarray(raw, float)
    n_raw = len(raw)
    out = [raw[0]]
    n = 1
    while n < n_raw:
        if raw[n] == raw[n - 1] and n + 1 < n_raw:
            out.extend([raw[n]] * (int(raw[n + 1]) + 1))
            n += 2
            if n < n_raw:               # value right after the count is a literal
                out.append(raw[n])
                n += 1
        else:
            out.append(raw[n])
            n += 1
    return np.cumsum(out)


def _time_scale(unit):
    """Multiplier that converts a DSV horizontal unit (HORIUNITNAME) to seconds."""
    u = unit.strip().lower()
    if "ms" in u:
        return 1e-3
    if "ns" in u:
        return 1e-9
    if u == "s":
        return 1.0
    if u.endswith("s"):                 # us, micro-sign + s in any encoding
        return 1e-6
    raise ValueError(f"Unrecognised DSV time unit '{unit}'; pass --dt_us")


def read_dsv(path, dt_us=None):
    """
    Read a Siemens IDEA/DSP-simulator .dsv file.

    Returns
    -----------------------
    g: 1D array, decompressed waveform times VERTFACTOR
    dt: sample interval in seconds (HORIDELTA, or dt_us if given)
    hdr: dict of header key/values
    """
    hdr, raw, in_vals = {}, [], False
    with open(path, encoding="latin-1") as f:      # latin-1 keeps the micro sign
        for line in f:
            s = line.strip()
            if not s or s.startswith(";"):          # ';' lines are comments
                continue
            if s.upper().startswith("[VALUES]"):
                in_vals = True
                continue
            if in_vals:
                if s.startswith("["):
                    break
                raw.append(float(s.split(";")[0].split()[0]))
            elif "=" in s:
                k, v = s.split("=", 1)
                hdr[k.strip()] = v.strip()
    if not raw:
        raise ValueError(f"No values found in {path}")

    g = _dsv_decompress(raw)
    if "SAMPLES" in hdr and len(g) != int(float(hdr["SAMPLES"])):
        n_hdr = int(float(hdr["SAMPLES"]))
        print(f"WARNING: {os.path.basename(path)}: decoded {len(g)} samples but header "
              f"says SAMPLES={n_hdr}")
        g = g[:n_hdr]
    g = g * float(hdr.get("VERTFACTOR", 1.0))

    if dt_us is not None:
        dt = dt_us * 1e-6
    elif "HORIDELTA" in hdr:
        dt = float(hdr["HORIDELTA"]) * _time_scale(hdr.get("HORIUNITNAME", "us"))
    else:
        raise ValueError(f"{path} has no HORIDELTA in its header; pass --dt_us")
    return g, dt, hdr


TERM_FOR_AXIS = {"x": 1, "y": 2, "z": 3}


def load_girf(path, axis, order=1, term=None):
    """
    Load a GIRF produced by the calibration script.

    `path` can be an .npz file, a folder containing
    SphericalHarmonics_<AXIS>_<order>.npz, a folder containing a subfolder
    SphericalHarmonics_<AXIS>_<order>/, or a folder containing
    freqFull.npy / GIRF_FT.npy / params.npy.

    Returns freq (Hz, increasing), H (complex 1D), params (dict).
    """
    term = TERM_FOR_AXIS[axis] if term is None else term
    if os.path.isdir(path):
        name = f"SphericalHarmonics_{axis.upper()}_{order}"
        npz = os.path.join(path, name + ".npz")
        sub = os.path.join(path, name)
        if os.path.exists(npz):
            path = npz
        elif os.path.isdir(sub):
            path = sub

    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        freq, girf, params = d["freqFull"], d["GIRF_FT"], d["params"].item()
    else:
        freq = np.load(os.path.join(path, "freqFull.npy"))
        girf = np.load(os.path.join(path, "GIRF_FT.npy"), allow_pickle=True)
        params = np.load(os.path.join(path, "params.npy"), allow_pickle=True)
        params = params.item() if params.shape == () else params

    if girf.ndim >= 2 and girf.shape[0] != len(freq):
        H = np.squeeze(girf[term])          # (nTerms, nFreq, 1) -> pick the term
    else:
        H = np.squeeze(girf)                # already a single term
    if H.ndim != 1 or len(H) != len(freq):
        raise ValueError(f"GIRF shape {girf.shape} does not match freq {freq.shape}")

    order_ = np.argsort(freq)
    return np.asarray(freq)[order_], H[order_], params


def save_mat(path, data):
    """
    Write a dict to .mat
    """
    savemat(path, data, oned_as='column')
