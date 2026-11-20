"""
Apply a measured GIRF to Siemens .dsv gradient waveforms and optionally
write nominal and GIRF-corrected MRSI trajectory CSV files.

Example:
  pixi run apply-girf --girf_dir /path/to/GIRF --order 1 \
    --dsv_x seq_GRX.dsv --dsv_y seq_GRY.dsv --dsv_z seq_GRZ.dsv \
    --out seq_GIRF.mat --nominal_kfile_out seq_nominal.csv \
    --girf_kfile_out seq_girf.csv --dwell_us 5 --dsv_adc seq_ADC.dsv \
    --n_tr 8 --kfile_plot_prefix seq_kspace
"""
import argparse
from pathlib import Path

import numpy as np

from io_helper import read_dsv, load_girf, save_mat, TERM_FOR_AXIS
from plotting import plot_kspace, plot_kfile_trajectories
from kfile import make_kfile, adc_window_from_dsv
from core import apply_girf, kspace, girf_dc_gain


def process_axis(axis, dsv_path, girf_path, order, term, dsv_scale, dt_us, pad_us):
    """Read one DSV, apply that axis's GIRF, and return MATLAB-ready arrays."""
    g, dt, hdr = read_dsv(dsv_path, dt_us)
    g = g * dsv_scale  # -> mT/m
    freq, H, params = load_girf(girf_path, axis, order, term)
    g_nom, g_act, clipped = apply_girf(g, dt, freq, H, pad_us)
    dc = abs(girf_dc_gain(H, freq))

    print(
        f"[{axis}] samples={len(g)} dt={dt*1e6:.3f} us "
        f"unit={hdr.get('VERTUNITNAME', '?')} band=+-{freq.max()/1e3:.1f} kHz "
        f"|H(0)|={dc:.3f} peak nom={np.abs(g_nom).max():.3f} "
        f"act={np.abs(g_act).max():.3f} mT/m"
    )
    if abs(dc - 1) > 0.1:
        print(f"    WARNING: |H(0)| is not ~1; check term index and units for axis {axis}")
    if clipped:
        print("    note: waveform extends above the measured GIRF band; those frequencies are set to 0")

    return {
        f"t_{axis}": np.arange(len(g_nom)) * dt,
        f"g{axis}_nom": g_nom,
        f"g{axis}_act": g_act,
        f"k{axis}_nom": kspace(g_nom, dt),
        f"k{axis}_act": kspace(g_act, dt),
        f"dt_{axis}": dt,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--girf_dir", help="Folder with SphericalHarmonics_<AXIS>_<order>.npz")
    ap.add_argument("--order", type=int, default=1)
    for axis in "xyz":
        ap.add_argument(f"--dsv_{axis}", help=f"DSV file for {axis.upper()} gradient")
        ap.add_argument(f"--girf_{axis}", help=f"GIRF .npz or folder for {axis.upper()} (overrides --girf_dir)")
        ap.add_argument(f"--term_{axis}", type=int, default=None,
                        help=f"GIRF term index for {axis.upper()} (default {TERM_FOR_AXIS[axis]})")
    ap.add_argument("--dsv_scale", type=float, default=1.0, help="Multiply DSV values to get mT/m")
    ap.add_argument("--dt_us", type=float, default=None, help="Override DSV sample interval (us)")
    ap.add_argument("--pad_us", type=float, default=500.0, help="Zero padding after waveform (us)")
    ap.add_argument("--out", required=True, help="Output .mat file")
    ap.add_argument("--plot", action="store_true", help="Show continuous gradient-integrated k-space plots")
    ap.add_argument("--save_fig", default=None, help="Save continuous k-space figure")
    ap.add_argument("--kfile_out", default=None,
                    help="Legacy alias for --girf_kfile_out (GIRF-predicted CSV)")
    ap.add_argument("--nominal_kfile_out", default=None, help="Write nominal ADC-sampled k-file (CSV)")
    ap.add_argument("--girf_kfile_out", default=None, help="Write GIRF-corrected ADC-sampled k-file (CSV)")
    ap.add_argument("--kfile_plot_prefix", default=None,
                    help="Save nominal, GIRF and overlay trajectory plots using this filename prefix")
    ap.add_argument("--dwell_us", type=float, default=None, help="ADC dwell time (us)")
    ap.add_argument("--n_adc", type=int, default=None, help="ADC samples per pass (default: from ADC DSV duration / dwell)")
    ap.add_argument("--adc_start_us", type=float, default=None, help="ADC start relative to DSV start (us)")
    ap.add_argument("--dsv_adc", default=None, help="*_ADC.dsv, used to infer ADC start and duration")
    ap.add_argument("--n_tr", type=int, default=1,
                    help="Number of TR rows to write; repeats the supplied one-pass trajectory")
    ap.add_argument("--k_time_unit", choices=["s", "ms", "us"], default="s",
                    help="Unit of the k-file time column")
    ap.add_argument("--k_absolute", action="store_true",
                    help="Keep absolute k instead of setting k=0 at the first ADC sample")
    return ap.parse_args()


def main():
    args = parse_args()
    out = {}
    for axis in "xyz":
        dsv = getattr(args, f"dsv_{axis}")
        if not dsv:
            continue
        girf_path = getattr(args, f"girf_{axis}") or args.girf_dir
        if not girf_path:
            raise SystemExit(f"Need --girf_dir or --girf_{axis} for axis {axis}")
        out.update(process_axis(
            axis, dsv, girf_path, args.order, getattr(args, f"term_{axis}"),
            args.dsv_scale, args.dt_us, args.pad_us
        ))
    if not out:
        raise SystemExit("Give at least one of --dsv_x / --dsv_y / --dsv_z")

    save_mat(args.out, out)
    print(f"Saved {args.out}")

    girf_path = args.girf_kfile_out or args.kfile_out
    nominal_path = args.nominal_kfile_out
    if nominal_path or girf_path or args.kfile_plot_prefix:
        if "kx_nom" not in out or "ky_nom" not in out:
            raise SystemExit("MRSI k-file generation requires both --dsv_x and --dsv_y")
        if args.dwell_us is None:
            raise SystemExit("K-file generation requires --dwell_us")
        adc_start = adc_duration = None
        if args.dsv_adc:
            adc_start, adc_duration = adc_window_from_dsv(args.dsv_adc)
            print(f"ADC DSV: start {adc_start*1e6:.3f} us, duration {adc_duration*1e6:.3f} us")
        if args.adc_start_us is not None:
            adc_start = args.adc_start_us * 1e-6
        if adc_start is None:
            raise SystemExit("K-file generation requires --adc_start_us or --dsv_adc")

        dwell = args.dwell_us * 1e-6
        n_adc = args.n_adc
        if n_adc is None and adc_duration is not None:
            n_adc = int(round(adc_duration / dwell))
        if not n_adc:
            raise SystemExit("K-file generation requires --n_adc or an ADC DSV with duration")

        common = (out, adc_start, dwell, n_adc, args.n_tr, args.k_time_unit,
                  not args.k_absolute)
        nominal_data = make_kfile(nominal_path, *common, kind="nom") if nominal_path else None
        girf_data = make_kfile(girf_path, *common, kind="act") if girf_path else None

        if args.kfile_plot_prefix:
            # Plotting can be requested without writing CSVs; build the data in memory.
            if nominal_data is None or girf_data is None:
                from kfile import build_kfile_data
                if nominal_data is None:
                    nominal_data = build_kfile_data(*common, kind="nom")
                if girf_data is None:
                    girf_data = build_kfile_data(*common, kind="act")
            plot_kfile_trajectories(nominal_data, girf_data,
                                    save_prefix=args.kfile_plot_prefix, show=args.plot)

    if args.plot or args.save_fig:
        plot_kspace(out, save=args.save_fig, show=args.plot)


if __name__ == "__main__":
    main()
