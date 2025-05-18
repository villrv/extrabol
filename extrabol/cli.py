# extrabol/cli.py

import os
import argparse
import numpy as np
from astropy.cosmology import Planck13 as cosmo

from .io import read_snana_file, read_in_photometry, write_output
from .interpolate import interpolate, test as test_template
from .blackbody import fit_bb, compute_bolometric_luminosity, compute_pseudo_bol
from .plotting import plot_gp, plot_bb_ev, plot_bb_bol

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Run EXTRABOL on a SNANA data file."
    )

    default_data = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "example", "SN2010bc.snana.dat"
    )

    parser.add_argument("snfile", nargs="?", default=default_data,
                        help="SNANA data file")
    parser.add_argument("--dm", type=float, default=None,
                        help="Distance modulus")
    parser.add_argument("--distance", "-d", type=float, default=None,
                        help="Luminosity distance (Mpc)")
    parser.add_argument("--z", "--redshift", dest="z", type=float,
                        default=None, help="Redshift")
    parser.add_argument("--mwebv", type=float, default=None,
                        help="Milky Way E(B–V)")
    parser.add_argument("--hostebv", type=float, default=0.0,
                        help="Host E(B–V)")
    parser.add_argument("--start", type=float, default=None,
                        help="Start phase (days)")
    parser.add_argument("--end", type=float, default=None,
                        help="End phase (days)")
    parser.add_argument("--snr", type=float, default=5.0,
                        help="S/N threshold")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("-m", "--mean", default="0",
                        help="Template mean: '1a','1bc','2p','2l','test', or '0'")
    parser.add_argument("--show-template", dest="show_template",
                        action="store_true",
                        help="Overlay template on GP plot")
    parser.add_argument("--outdir", default="products",
                        help="Output directory")
    parser.add_argument("--stepsize", type=float, default=0.5,
                        help="GP time step (days)")
    parser.add_argument("--kernel-width", type=float, nargs=2,
                        help="Initial GP kernel width")
    parser.add_argument("--mc", action="store_true", dest="use_mcmc",
                        help="Use MCMC for blackbody fits")
    parser.add_argument("--T-max", type=float, default=None,
                        help="Max temperature for BB fit")

    parser.add_argument('--pseudo', dest='pseudo',
                        action='store_true',
                        help='Skip blackbody fits and compute pseudo‐bolometric LC '
                             '(integrated flux over filters).')

    opts = parser.parse_args(args=args)

    # ensure outdir ends in slash
    if not opts.outdir.endswith(os.sep):
        opts.outdir += os.sep
    os.makedirs(opts.outdir, exist_ok=True)

    # read SNANA metadata (or fall back to cmd‐line z, E(B–V))
    if opts.snfile.endswith((".snana.dat", ".snana")):
        metadata, _ = read_snana_file(opts.snfile)
        file_z   = float(metadata.get("REDSHIFT", 0.0))
        file_ebv = float(metadata.get("MWEBV",   0.0))
    else:
        file_z, file_ebv = opts.z or 0.0, opts.mwebv or 0.0

    redshift = opts.z if opts.z is not None else file_z
    mwebv    = opts.mwebv if opts.mwebv is not None else file_ebv

    # distance modulus
    if opts.dm is not None:
        dm = opts.dm
    elif opts.distance is not None:
        dm = 5.0 * np.log10(opts.distance * 1e5)
    else:
        dm = cosmo.distmod(redshift).value

    # read & preprocess
    lc, wv_corr, flux_corr, filters = read_in_photometry(
        opts.snfile, dm=dm, redshift=redshift,
        start=opts.start, end=opts.end,
        snr=opts.snr, mwebv=mwebv,
        hostebv=opts.hostebv, verbose=opts.verbose
    )

    # template selection
    if opts.mean == "test":
        opts.mean = test_template(lc, wv_corr, redshift)
    if opts.verbose:
        print(f"Using template: {opts.mean}")

    # GP interpolation
    dense_lc, test_data, test_times, dense_times = interpolate(
        lc,
        wv_corr,
        flux_corr,
        sn_type=opts.mean,
        use_mean=(opts.mean != "0"),
        z=redshift,
        verbose=opts.verbose,
        stepsize=opts.stepsize,
        kernel_width=opts.kernel_width
    )

    # undo the preprocessing flux shift
    dense_lc[:, :, 0] += flux_corr

    # prepare for BB fits
    base = os.path.basename(opts.snfile)
    snname = os.path.splitext(os.path.splitext(base)[0])[0]
    wv_norms, wv_inds = np.unique(lc.T[:,2], return_index=True)
    wvs = wv_norms * 1000.0 + wv_corr
    filters_unique = [filters[i] for i in wv_inds]

    # blackbody or pseudo–bol
    if not opts.pseudo:
        T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr = fit_bb(
            dense_lc, wvs,
            use_mcmc=opts.use_mcmc,
            T_max=opts.T_max
        )
        bol_lum, bol_err = compute_bolometric_luminosity(
            T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr
        )
    else:
        # no BB arrays needed
        T_arr = R_arr = Terr_arr = Rerr_arr = covar_arr = None
        bol_lum, bol_err = compute_pseudo_bol(
            dense_lc,      # T×F×2
            wvs,           # filter wavelengths [Å]
        )

    # always plot the GP + data + (optional) template
    plot_gp(
        lc, dense_times, dense_lc,
        snname, flux_corr, filters_unique, wvs,
        test_data, opts.outdir,
        opts.mean, test_times,
        (opts.mean != "0"), opts.show_template
    )

    # only do BB‐evolution / BB‐bol if real BB fits were run
    if not opts.pseudo:
        plot_bb_ev(dense_times, T_arr, R_arr, Terr_arr, Rerr_arr,
                   snname, opts.outdir, opts.mean)
    plot_bb_bol(dense_times, bol_lum, bol_err,
                    snname, opts.outdir, opts.mean, opts.pseudo)

    # write out tabular results (will include either BB or pseudo‐bol columns)
    write_output(
        lc, dense_times, dense_lc,
        T_arr, Terr_arr, R_arr, Rerr_arr,
        bol_lum, bol_err, filters_unique,
        snname, opts.outdir, opts.mean, opts.pseudo
    )

    print("All done!")

if __name__ == "__main__":
    main()
