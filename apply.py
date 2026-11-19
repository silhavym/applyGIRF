
"""
Apply a measured GIRF to Siemens .dsv gradient waveform and save a MATLAB .mat file with the results.
 
With pixi:
    pixi run apply-girf --girf_dir /path/to/GIRF --order 1 --dsv_x seq_GRX.dsv --dsv_y seq_GRY.dsv --dsv_z seq_GRZ.dsv --out seq_GIRF.mat --plot
"""
import argparse
from io_helper import read_dsv, load_girf, save_mat, TERM_FOR_AXIS
from plotting import plot_kspace
from kfile import make_kfile, adc_window_from_dsv
from core import apply_girf, kspace, girf_dc_gain 
import numpy as np
 
def process_axis(axis, dsv_path, girf_path, order, term, dsv_scale, dt_us, pad_us):
    """Read one DSV, apply that axis's GIRF, return a dict of MATLAB variables."""
    g, dt, hdr = read_dsv(dsv_path, dt_us)
    g = g * dsv_scale                                   # -> mT/m
    freq, H, params = load_girf(girf_path, axis, order, term)
 
    g_nom, g_act, clipped = apply_girf(g, dt, freq, H, pad_us)
    dc = abs(girf_dc_gain(H, freq))
 
    print(f"[{axis}] samples={len(g)} dt={dt*1e6:.3f} us  unit={hdr.get('VERTUNITNAME', '?')}  band=+-{freq.max()/1e3:.1f} kHz  "
          f"|H(0)|={dc:.3f}  peak nom={np.abs(g_nom).max():.3f} act={np.abs(g_act).max():.3f} mT/m")
    if abs(dc - 1) > 0.1:
        print(f"    WARNING: |H(0)| is not ~1 -> check term index and units for axis {axis}")
    if clipped:
        print("    note: waveform extends above the measured GIRF band; those frequencies are set to 0")
 
    return {
        f"t_{axis}": np.arange(len(g_nom)) * dt,        # s
        f"g{axis}_nom": g_nom,                          # mT/m
        f"g{axis}_act": g_act,                          # mT/m
        f"k{axis}_nom": kspace(g_nom, dt),              # cycles/m
        f"k{axis}_act": kspace(g_act, dt),              # cycles/m
        f"dt_{axis}": dt,
    }
 
 
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--girf_dir", help="Folder with SphericalHarmonics_<AXIS>_<order>.npz")
    ap.add_argument("--order", type=int, default=1)
    for a in "xyz":
        ap.add_argument(f"--dsv_{a}", help=f"DSV file for {a.upper()} gradient")
        ap.add_argument(f"--girf_{a}", help=f"GIRF .npz or folder for {a.upper()} (overrides --girf_dir)")
        ap.add_argument(f"--term_{a}", type=int, default=None,
                        help=f"GIRF term index for {a.upper()} (default {TERM_FOR_AXIS[a]})")
    ap.add_argument("--dsv_scale", type=float, default=1.0, help="Multiply DSV values to get mT/m")
    ap.add_argument("--dt_us", type=float, default=None, help="Override DSV sample interval (us)")
    ap.add_argument("--pad_us", type=float, default=500.0, help="Zero padding after waveform (us)")
    ap.add_argument("--out", required=True, help="Output .mat file")
    ap.add_argument("--plot", action="store_true", help="Show k-space before/after GIRF")
    ap.add_argument("--save_fig", default=None, help="Save the k-space figure (e.g. kspace.png)")
    ap.add_argument("--kfile_out", default=None, help="Write a k-file (CSV: TR,Kx,Ky,time) from the GIRF-predicted k-space (needs x and y)")
    ap.add_argument("--dwell_us", type=float, default=None, help="ADC dwell time (us), needed for --kfile_out")
    ap.add_argument("--n_adc", type=int, default=None, help="ADC samples per pass (default: from --dsv_adc duration / dwell)")
    ap.add_argument("--adc_start_us", type=float, default=None, help="ADC start relative to DSV start (us)")
    ap.add_argument("--dsv_adc", default=None, help="*_ADC.dsv, gives ADC start (and length) if not set explicitly")
    ap.add_argument("--n_tr", type=int, default=1, help="Write the one-pass trajectory for this many TRs (default 1)")
    ap.add_argument("--k_time_unit", choices=["s", "ms", "us"], default="s", help="Unit of the 'time' column (default s)")
    ap.add_argument("--k_absolute", action="store_true", help="Keep absolute k instead of setting k=0 at the first ADC sample")
    return ap.parse_args()
 
 
def main():
    args = parse_args()
    out = {}
    for a in "xyz":
        dsv = getattr(args, f"dsv_{a}")
        if not dsv:
            continue
        girf_path = getattr(args, f"girf_{a}") or args.girf_dir
        if not girf_path:
            raise SystemExit(f"Need --girf_dir or --girf_{a} for axis {a}")
        out.update(process_axis(a, dsv, girf_path, args.order, getattr(args, f"term_{a}"),
                                args.dsv_scale, args.dt_us, args.pad_us))
    if not out:
        raise SystemExit("Give at least one of --dsv_x / --dsv_y / --dsv_z")
    save_mat(args.out, out)
    print(f"Saved {args.out}")
    if args.kfile_out:
        if "kx_act" not in out or "ky_act" not in out:
            raise SystemExit("--kfile_out needs both --dsv_x and --dsv_y")
        if args.dwell_us is None:
            raise SystemExit("--kfile_out needs --dwell_us")
        start = length = None
        if args.dsv_adc:
            start, length = adc_window_from_dsv(args.dsv_adc)
            print(f"ADC from {args.dsv_adc}: start {start*1e6:.1f} us, duration {length*1e6:.1f} us")
        if args.adc_start_us is not None:
            start = args.adc_start_us * 1e-6
        if start is None:
            raise SystemExit("--kfile_out needs --adc_start_us or --dsv_adc")
        dwell = args.dwell_us * 1e-6
        n_adc = args.n_adc if args.n_adc else (int(length / dwell + 1e-6) if length else None)
        if not n_adc:
            raise SystemExit("--kfile_out needs --n_adc or --dsv_adc")
        make_kfile(args.kfile_out, out, start, dwell, n_adc, args.n_tr, args.k_time_unit,
                   not args.k_absolute)
    if args.plot or args.save_fig:
        plot_kspace(out, save=args.save_fig, show=args.plot)
 
 
if __name__ == "__main__":
    main()