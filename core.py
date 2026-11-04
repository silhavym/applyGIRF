"""
The math: applies GIRF to a gradient waveform and integrate to k-space
"""
import numpy as np 

GAMMA = 42.576e3  # Hz/T

def apply_girf(g_nom, dt, freq, H, pad_us=500.0):
    """
    Apply GIRF to nomial gradient waveform
    """
    n_out = len(g_nom) + int(round(pad_us *1e-6 / dt))
    nnft = int(2 ** np.ceil(np.log2(2*n_out)))

    f = np.fft.rfftfreq(nnft, dt)

    Hf = (np.interp(f, freq, H.real, left=0, right=0) + 1j * np.interp(f, freq, H.imag, left=0, right=0))
    g_act = np.fft.irfft(np.fft.rfft(g_nom, nnft) * Hf, nnft)[:n_out]
    g_pad = np.zeros(n_out)
    g_pad[:len(g_nom)] = g_nom

    band_exceeded = f.max() > freq.max() + 2 * (freq[1] - freq[0])
    return g_pad, g_act, band_exceeded

def kspace(g,dt):
    """
    Integrate gradient waveform to k-space
    """
    k = np.cumsum(g) * dt * GAMMA
    return k

def girf_dc_gain(H, freq):
    """
    Compute the DC gain of a GIRF
    """
    return np.interp(0, freq, H.real) + 1j * np.interp(0, freq, H.imag)

