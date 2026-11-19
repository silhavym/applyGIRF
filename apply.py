"""
Apply a measured GIRF to Siemens .dsv gradient waveform and save a MATLAB .mat file
plus separate nominal and GIRF-adapted 2D k-files (tr Kx Ky).

With pixi:
    pixi run apply-girf --girf_dir /path/to/GIRF --order 1 \
  --dsv_x seq_GRX.dsv --dsv_y seq_GRY.dsv --dsv_z seq_GRZ.dsv \
  --dsv_rfd seq_RFD.dsv --dsv_rfp seq_RFP.dsv --dsv_adc seq_ADC.dsv \
  --out seq_GIRF.mat \
  --kfile seq.kfile \
  --save_fig kspace.png
"""
import argparse
import numpy as np
from io_helper import read_dsv, read_adc_dsv, load_girf, save_mat, TERM_FOR_AXIS
from plotting import plot_kspace
from kfile import build_kfile, write_kfile, resolve_kfile_paths
from core import apply_girf, kspace, girf_dc_gain


def process_axis(axis, dsv_path, girf_path, order, term, dsv_scale, dt_us, pad_us):
    """Read one DSV, apply that axis's GIRF, return a dict of MATLAB variables."""
    g, dt, _ = read_dsv(dsv_path, dt_us)
    g = g * dsv_scale  # -> mT/m
    freq, H, _ = load_girf(girf_path, axis, order, term)

    g_nom, g_act, clipped = apply_girf(g, dt, freq, H, pad_us)
    dc = abs(girf_dc_gain(H, freq))

    print(
        f"[{axis}] samples={len(g)} dt={dt*1e6:.3f} us  band=+-{freq.max()/1e3:.1f} kHz  "
        f"|H(0)|={dc:.3f}  peak nom={np.abs(g_nom).max():.3f} act={np.abs(g_act).max():.3f} mT/m"
    )
    if abs(dc - 1) > 0.1:
        print(f"    WARNING: |H(0)| is not ~1 -> check term index and units for axis {axis}")
    if clipped:
        print("    note: waveform extends above the measured GIRF band; those frequencies are set to 0")

    return {
        f"t_{axis}": np.arange(len(g_nom)) * dt,  # s
        f"g{axis}_nom": g_nom,                    # mT/m
        f"g{axis}_act": g_act,                    # mT/m
        f"k{axis}_nom": kspace(g_nom, dt),        # cycles/m
        f"k{axis}_act": kspace(g_act, dt),        # cycles/m
        f"dt_{axis}": dt,
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--girf_dir", help="Folder with SphericalHarmonics_<AXIS>_<order>.npz")
    ap.add_argument("--order", type=int, default=1)
    for a in "xyz":
        ap.add_argument(f"--dsv_{a}", help=f"DSV file for {a.upper()} gradient")
        ap.add_argument(f"--girf_{a}", help=f"GIRF .npz or folder for {a.upper()} (overrides --girf_dir)")
        ap.add_argument(
            f"--term_{a}",
            type=int,
            default=None,
            help=f"GIRF term index for {a.upper()} (default {TERM_FOR_AXIS[a]})",
        )
    ap.add_argument("--dsv_rfd", default=None, help="Optional RF amplitude/magnitude DSV file")
    ap.add_argument("--dsv_rfp", default=None, help="Optional RF phase DSV file")
    ap.add_argument("--dsv_adc", default=None, help="ADC DSV file specifying ADC sample points")
    ap.add_argument("--dsv_scale", type=float, default=1.0, help="Multiply DSV values to get mT/m")
    ap.add_argument("--dt_us", type=float, default=None, help="Override DSV sample interval (us)")
    ap.add_argument("--pad_us", type=float, default=500.0, help="Zero padding after waveform (us)")
    ap.add_argument("--out", required=True, help="Output .mat file")
    ap.add_argument("--kfile", default=None, help="Base path to write <base>_nom.kfile and <base>_girf.kfile")
    ap.add_argument("--adc_start_us", type=float, default=0.0, help="Optional offset added to ADC times (us)")
    ap.add_argument("--adc_n", type=int, default=None, help="Optional max number of ADC samples")
    ap.add_argument("--points_per_leaf", type=int, default=176, help="ADC samples per rosette leaf (default: 176)")
    ap.add_argument("--n_leaves", type=int, default=None, help="Optional max rosette leaf count")
    ap.add_argument("--plot", action="store_true", help="Show k-space before/after GIRF")
    ap.add_argument("--save_fig", default=None, help="Save the k-space figure (e.g. kspace.png)")
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
        out.update(
            process_axis(
                a,
                dsv,
                girf_path,
                args.order,
                getattr(args, f"term_{a}"),
                args.dsv_scale,
                args.dt_us,
                args.pad_us,
            )
        )
    if not out:
        raise SystemExit("Give at least one of --dsv_x / --dsv_y / --dsv_z")

    if args.dsv_rfd:
        rfd, dt_rfd, _ = read_dsv(args.dsv_rfd, args.dt_us)
        out["rf_mag"] = rfd
        out["dt_rfd"] = dt_rfd
    if args.dsv_rfp:
        rfp, dt_rfp, _ = read_dsv(args.dsv_rfp, args.dt_us)
        out["rf_phase"] = rfp
        out["dt_rfp"] = dt_rfp

    save_mat(args.out, out)
    print(f"Saved {args.out}")

    if args.kfile:
        if not args.dsv_adc:
            raise SystemExit("--kfile requires --dsv_adc to read ADC sample points from the DSV")
        t_adc_s = read_adc_dsv(args.dsv_adc, dt_us=args.dt_us)
        nom_path, girf_path = resolve_kfile_paths(args.kfile)
        for kind, path in (("nom", nom_path), ("girf", girf_path)):
            kf = build_kfile(
                out,
                kind=kind,
                t_adc_s=t_adc_s,
                n_adc=args.adc_n,
                start_us=args.adc_start_us,
                points_per_leaf=args.points_per_leaf,
                n_leaves=args.n_leaves,
            )
            write_kfile(path, kf)
            print(f"Saved {path}  ({len(kf['t_adc'])} ADC points, kind={kind})")

    if args.plot or args.save_fig:
        plot_kspace(out, save=args.save_fig, show=args.plot)


if __name__ == "__main__":
    main()