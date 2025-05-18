# extrabol/io.py

import os
import sys
import re
import numpy as np
import pandas as pd
import importlib_resources
import extinction
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
from astropy.io import ascii


def read_snana_file(file_path):
    """
    Reads a SNANA-format file and returns (metadata dict, pandas.DataFrame).
    """
    metadata = {}
    varlist = []
    data_lines = []
    header = False

    with open(file_path, "r") as f:
        for line in f:
            txt = line.strip()
            if txt.startswith("#"):
                continue
            if txt.startswith("NOBS"):
                header = True
                continue

            if not header:
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip().split("+-")[0]
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
                    metadata[key.strip()] = nums[0] if nums else val
            else:
                if line.startswith("VARLIST:"):
                    varlist = line.split()[1:]
                elif line.startswith("OBS:"):
                    data_lines.append(line.split()[1:])

    df = pd.DataFrame(data_lines, columns=varlist)
    # Convert numeric columns to float, keep FLT & MAGTYPE as str
    for col in varlist:
        if col not in ("FLT", "MAGTYPE"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return metadata, df


def read_in_photometry(filename, dm, redshift,
                       start, end, snr,
                       mwebv, hostebv, verbose):
    """
    Read in photometry (SNANA or ASCII), apply cuts, return:
      lc        : np.ndarray shape (5,N) [phase, flux, wv_norm, err, width]
      wv_corr   : float
      flux_corr : float
      filters   : list of str
    """

    # ─── guard None start/end ───
    if start is None:
        start = -np.inf
    if end is None:
        end = np.inf

    # ─── load data ─────────────────
    if ".snana" in filename:
        if verbose:
            print("Assuming SNANA format")
        _, df = read_snana_file(filename)
        phases = df["MJD"].values.astype(float)
        errs    = df["MAGERR"].values.astype(float)
        flt     = df["FLT"].values.astype(str)
        mags    = df["MAG"].values.astype(float)
        mtype   = df["MAGTYPE"].values.astype(str)
    else:
        # fallback for pure ASCII
        data = np.loadtxt(filename, dtype=str, skiprows=2)
        phases = data[:,0].astype(float)
        mags    = data[:,1].astype(float)
        errs    = data[:,2].astype(float)
        flt     = data[:,3].astype(str)
        mtype   = data[:,-1].astype(str)

    # ─── distance modulus ──────────
    if dm is None:
        dL_pc = cosmo.luminosity_distance(redshift).to_value("pc")
        dm = 5.0 * np.log10(dL_pc) - 5.0
        if verbose:
            print(f"Computed DM={dm:.3f} from z={redshift:.3f}")

    # ─── filter metadata ───────────
    pkg = importlib_resources.files("extrabol.filter_data")
    fps = Table.read(pkg/"fps.xml")
    IDs  = fps["filterID"].astype(str)
    lam_eff   = fps["WavelengthEff"].astype(float)
    width_eff = fps["WidthEff"].astype(float)
    zpt_all   = fps["ZeroPoint"].astype(str)

    # map observed filters to their effective wavelengths
    wv_effs, widths, my_filters = [], [], []
    for f in flt:
        idx = np.where(IDs==f)[0]
        if not idx.size:
            sys.exit(f"Cannot find filter {f} in fps.xml")
        wv_effs.append(lam_eff[idx[0]])
        widths.append(width_eff[idx[0]])
        my_filters.append(f)
    wv_effs = np.array(wv_effs)
    widths  = np.array(widths)

    # ─── mags → GP-friendly fluxes ─
    fluxes = []
    for m, mt, f in zip(mags, mtype, flt):
        m_corr = m - dm + 2.5*np.log10(1.0+redshift)
        if mt=="AB":
            zp = 0.0
        else:
            zi = np.where(IDs==f)[0][0]
            zp = 2.5*np.log10(float(zpt_all[zi]) / 3631.0)
        fluxes.append(zp - m_corr)
    fluxes = np.array(fluxes)

    # ─── remove extinction ─────────
    ext_mw   = extinction.fm07(wv_effs, mwebv*3.1)
    ext_host = extinction.fm07(wv_effs/(1+redshift), hostebv*3.1)
    fluxes  += (ext_mw + ext_host)

    # normalize for GP (center), *and* convert Å → 1000Å units
    wv_corr   = np.mean(wv_effs/(1+redshift))
    flux_corr = np.min(fluxes) - 1.0
    # original code did wv_effs/(1+z) - wv_corr, then divided by 1000:
    wv_norm   = ((wv_effs/(1+redshift)) - wv_corr) / 1000.0
    fluxes   -= flux_corr

    # ─── apply SNR cut ─────────────
    keep = (1.0/errs) >= snr
    phases, fluxes, wv_norm, errs, widths, my_filters = (
        phases[keep], fluxes[keep], wv_norm[keep],
        errs[keep], widths[keep], np.array(my_filters)[keep]
    )

    # ─── shift peak to phase=0 ─────
    peak = np.argmax(fluxes)
    if verbose:
        print("Peak Luminosity occurs at MJD", phases[peak])
    phases = (phases - phases[peak])/(1+redshift)

    # ─── time window cut ───────────
    mask = (phases>=start)&(phases<=end)
    phases, fluxes, wv_norm, errs, widths, my_filters = (
        phases[mask], fluxes[mask], wv_norm[mask],
        errs[mask], widths[mask], my_filters[mask]
    )

    lc = np.vstack((phases, fluxes, wv_norm, errs, widths))
    return lc, wv_corr, flux_corr, list(my_filters)


def write_output(lc, dense_times, dense_lc,
                 Tarr, Terr_arr, Rarr, Rerr_arr,
                 bol_lum, bol_err, ufilts,
                 snname, outdir, sn_type, pseudo):
    """
    Write out the interpolated LC and BB (or pseudo‐bolometric) information
    as an ASCII table.
    """
    import os
    import pandas as pd
    from astropy.table import Table
    from astropy.io import ascii

    # Build a pandas DataFrame
    data = {"Phase": dense_times}
    for j, filt in enumerate(ufilts):
        data[filt]         = dense_lc[:, j, 0]
        data[filt + "_err"] = dense_lc[:, j, 1]
    data["Temp. (K)"]         = Tarr
    data["Temp. Err. (K)"]    = Terr_arr
    data["Radius (cm)"]       = Rarr
    data["Radius Err. (cm)"]  = Rerr_arr
    data["Bol. Lum. (erg/s)"] = bol_lum
    data["Bol. Err. (erg/s)"] = bol_err

    df = pd.DataFrame(data)
    table = Table.from_pandas(df)

    # flip sign on flux columns to match original
    for filt in ufilts:
        table[filt] *= -1.0

    if pseudo:
        tag = f"{snname}_pseudo_{sn_type}.txt"
    else:
        tag = f"{snname}_{sn_type}.txt"

    outname = os.path.join(outdir, tag)

    ascii.write(
        table,
        outname,
        formats={
            "Phase":        "%0.3f",
            **{filt:         "%0.3f" for filt in ufilts},
            **{filt + "_err": "%0.3f" for filt in ufilts},
            "Temp. (K)":         "%0.3e",
            "Temp. Err. (K)":    "%0.3e",
            "Radius (cm)":       "%0.3e",
            "Radius Err. (cm)":  "%0.3e",
            "Bol. Lum. (erg/s)": "%0.3e",
            "Bol. Err. (erg/s)": "%0.3e",
        },
        overwrite=True
    )

    return 1
