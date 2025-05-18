# extrabol/plotting.py

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_gp(lc, dense_times, dense_lc,
            snname, flux_corr, filters, wvs,
            test_data, outdir,
            sn_type, test_times,
            mean, show_template):
    """
    Plot the GP-interpolated light curves with optional template overlay,
    plus the original photometry points.
    """
    # lc shape: (5, N) — rows are: phase, flux, wv_norm, err, width
    phases, fluxes, wv_norms, errs, _ = lc

    # Unique normalized wavelengths, in same order as `filters`
    u_wv = np.unique(wv_norms)

    # Build color map keyed by filter name
    cmap = plt.get_cmap('rainbow')
    norm = (wvs - wvs.min()) / (wvs.max() - wvs.min())
    colors = {filters[i]: cmap(norm[i]) for i in range(len(filters))}

    plt.figure(figsize=(8, 5))

    # 1) GP curves + uncertainty shading
    for i, filt in enumerate(filters):
        col = colors[filt]
        gp_mag = -dense_lc[:, i, 0]      # dense_lc already has flux_corr added back
        gp_err = dense_lc[:, i, 1]
        plt.plot(dense_times, gp_mag, color=col, lw=1.8, label=filt)
        plt.fill_between(dense_times,
                         gp_mag - gp_err,
                         gp_mag + gp_err,
                         color=col, alpha=0.3)

    # 2) Template overlay
    if mean and show_template and test_data is not None:
        for i, filt in enumerate(filters):
            col = colors[filt]
            plt.plot(test_times,
                     -(test_data[i, :] + flux_corr),
                     '--', color=col, lw=1.2)

    # 3) Original photometry points + error bars
    for i, filt in enumerate(filters):
        col = colors[filt]
        mask = wv_norms == u_wv[i]
        x = phases[mask]
        y = -fluxes[mask] - flux_corr   # add back flux_corr for correct brightness
        yerr = errs[mask]
        plt.plot(x, y, 'o', color=col, ms=6, mec='k', mew=0.5)
        plt.errorbar(x, y, yerr=yerr,
                     fmt='none', color=col, capsize=2)

    # Final touches
    plt.title(f"{snname} Light Curves")
    plt.xlabel("Time (days)")
    plt.ylabel("Absolute Magnitudes")
    plt.gca().invert_yaxis()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Filter", loc="best")
    plt.tight_layout()

    # Save figure
    fname = os.path.join(outdir, f"{snname}_{sn_type}_gp.png")
    plt.savefig(fname)
    plt.close()


def plot_bb_ev(dense_times, Tarr, Rarr, Terr_arr, Rerr_arr,
               snname, outdir, sn_type):
    """
    Plot temperature and radius evolution.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    ax1.plot(dense_times, Tarr / 1e3, '-k')
    ax1.fill_between(dense_times,
                     (Tarr - Terr_arr) / 1e3,
                     (Tarr + Terr_arr) / 1e3,
                     color='gray', alpha=0.3)
    ax1.set_ylabel("Temp. (1000 K)")
    ax1.set_title(f"{snname} Blackbody Evolution")

    ax2.plot(dense_times, Rarr / 1e15, '-k')
    ax2.fill_between(dense_times,
                     (Rarr - Rerr_arr) / 1e15,
                     (Rarr + Rerr_arr) / 1e15,
                     color='gray', alpha=0.3)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel(r"Radius ($10^{15}$ cm)")

    plt.tight_layout()
    fname = os.path.join(outdir, f"{snname}_{sn_type}_bb_ev.png")
    plt.savefig(fname)
    plt.close()


def plot_bb_bol(dense_times, bol_lum, bol_err,
                snname, outdir, sn_type, pseudo=False):
    """
    Plot (pseudo-)bolometric luminosity evolution.

    Parameters
    ----------
    dense_times : array-like
        Time grid (days).
    bol_lum : array-like
        Bolometric luminosity (erg/s).
    bol_err : array-like
        1σ error on bolometric luminosity.
    snname : str
        Supernova name.
    outdir : str
        Output directory (will be created if needed).
    sn_type : str
        Template type (or '0').
    pseudo : bool, optional
        If True, indicates this is the pseudo-bolometric curve
        and appends "_pseudo" to the output filename.
    """
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(dense_times, bol_lum, '-k', label='Bolometric L')
    plt.fill_between(dense_times,
                     bol_lum - bol_err,
                     bol_lum + bol_err,
                     color='gray', alpha=0.3)
    plt.yscale('log')
    plt.xlabel("Time (days)")
    plt.ylabel("Bolometric Luminosity (erg/s)")
    subtitle = "Pseudo-" if pseudo else ""
    plt.title(f"{snname} {subtitle}Bolometric Luminosity")
    plt.tight_layout()

    tag = "_pseudo" if pseudo else ""
    fname = os.path.join(outdir,
                         f"{snname}_{sn_type}_bb_bol{tag}.png")
    plt.savefig(fname)
    plt.close()
