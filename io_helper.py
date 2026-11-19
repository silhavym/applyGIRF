"""
Input/output helpers: Siemens DSV reader, GIRF loader, MATLAB .mat writer
"""

import os
import numpy as np
from scipy.io import savemat


def read_dsv(path, dt_us=None):
    """
    Read a Siemens .dsv file.

    Returns
    -----------------------
    g: 1D array, values scaled by VERTFACTOR
    dt: sample interval in seconds
    hdr: dict of header key/values
    """
    hdr, vals, in_vals = {}, [], False
    with open(path, errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("[VALUES]"):
                in_vals = True
                continue
            if in_vals:
                if s.startswith("["):
                    break
                vals.append(float(s.split()[0]))
            elif "=" in s:
                k, v = s.split("=", 1)
                hdr[k.strip()] = v.strip()
    if not vals:
        raise ValueError(f"No values found in {path}")

    g = np.array(vals) * float(hdr.get("VERTFACTOR", 1.0))

    if dt_us is None:
        dt = float(hdr.get("VERTINTERVAL", 1.0)) * 1e-6
    else:
        dt = dt_us * 1e-6
    return g, dt, hdr


def read_adc_dsv(path, dt_us=None):
    """
    Read an ADC .dsv file and return ADC sample times in seconds.
    """
    adc_vals, dt, _ = read_dsv(path, dt_us=dt_us)
    active_idx = np.flatnonzero(adc_vals != 0)
    if len(active_idx) == 0:
        raise ValueError(f"No active ADC samples found in {path}")
    return active_idx.astype(float) * dt


TERM_FOR_AXIS = {"x": 1, "y": 2, "z": 3}


def load_girf(path, axis, order=1, term=None):
    """
    Load a GIRF produced by the calibration script.

    `path` can be an .npz file, a folder containing
    SphericalHarmonics_<AXIS>_<order>.npz, or a folder containing
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
        H = np.squeeze(girf[term])  # (nTerms, nFreq, 1) -> pick the term
    else:
        H = np.squeeze(girf)  # already a single term
    if H.ndim != 1 or len(H) != len(freq):
        raise ValueError(f"GIRF shape {girf.shape} does not match freq {freq.shape}")

    order_ = np.argsort(freq)
    return np.asarray(freq)[order_], H[order_], params


def save_mat(path, data):
    """
    Write a dict to .mat
    """
    savemat(path, data, oned_as="column")