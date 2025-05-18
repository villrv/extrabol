# extrabol/blackbody.py

import numpy as np
from scipy.optimize import curve_fit
import emcee

# Physical constants
c = 2.99792458e10          # speed of light [cm/s]
h = 6.62607015e-27         # Planck constant [erg·s]
k_B = 1.380649e-16         # Boltzmann constant [erg/K]
ang_to_cm = 1e-8           # Å → cm
sigsb = 5.670374419e-5     # Stefan–Boltzmann [erg/cm²/K⁴/s]

def bbody(lam, T, R):
    """
    Blackbody spectral luminosity L_lambda at wavelength lam (Å),
    temperature T (K), radius R (cm).
    """
    lam_cm = lam * ang_to_cm
    exponential = (h * c) / (lam_cm * k_B * T)
    blam = ((2.0 * np.pi * h * c**2) / lam_cm**5) / (np.exp(exponential) - 1.0)
    area = 4.0 * np.pi * R**2
    return blam * area

def fit_bb(dense_lc, wvs, use_mcmc, T_max):
    """
    Fit blackbody temperature and radius to each epoch of the dense light curve.

    Returns arrays T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr.
    """
    n_epochs = len(dense_lc)
    T_arr = np.zeros(n_epochs)
    R_arr = np.zeros(n_epochs)
    Terr_arr = np.zeros(n_epochs)
    Rerr_arr = np.zeros(n_epochs)
    covar_arr = np.zeros(n_epochs)

    # Initial guess for curve_fit
    prior_fit = (9000, 1e15)

    for i, epoch in enumerate(dense_lc):
        # Convert magnitude‐like values back to flux units:
        fnu = 10**((-epoch[:,0] + 48.6) / -2.5)
        ferr = epoch[:,1]
        fnu = fnu * 4.0 * np.pi * (3.086e19)**2
        fnu_err = np.abs(0.921034 * 10**(0.4 * epoch[:,0] - 19.44)) \
                   * ferr * 4.0 * np.pi * (3.086e19)**2

        # Convert to spectral flux
        flam = fnu * c / (wvs * ang_to_cm)**2
        flam_err = fnu_err * c / (wvs * ang_to_cm)**2

        if use_mcmc:
            def log_prior(params):
                T, R = params
                if T > 0 and (T_max is None or T < T_max) and R > 0:
                    return 0.0
                return -np.inf

            def log_likelihood(params, lam, f, f_err):
                model = bbody(lam, *params)
                return -0.5 * np.sum(((f - model) / f_err)**2)

            def log_posterior(params, lam, f, f_err):
                lp = log_prior(params)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood(params, lam, f, f_err)

            ndim, nwalkers = 2, 16
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_posterior, args=(wvs, flam, flam_err)
            )
            p0 = np.vstack((np.random.normal(prior_fit[0], 1000, nwalkers),
                            np.random.normal(prior_fit[1], 1e14, nwalkers))).T
            sampler.run_mcmc(p0, 500, progress=False)
            samples = sampler.get_chain(discard=100, flat=True)
            T_arr[i] = np.median(samples[:,0])
            R_arr[i] = np.median(samples[:,1])
            Terr_arr[i] = np.std(samples[:,0])
            Rerr_arr[i] = np.std(samples[:,1])
            covar_arr[i] = np.cov(samples.T)[0,1]

        else:
            # Standard curve_fit
            upper_T = T_max if (T_max is not None) else 1e6
            bounds = (0, [upper_T, np.inf])

            try:
                BBparams, pcov = curve_fit(
                    bbody, wvs, flam,
                    p0=prior_fit,
                    sigma=flam_err,
                    bounds=bounds,
                    maxfev=10000,
                    method='dogbox',
                    absolute_sigma=True
                )
                T_arr[i], R_arr[i] = BBparams
                Terr_arr[i], Rerr_arr[i] = np.sqrt(np.diag(pcov))
                covar_arr[i] = pcov[0,1]
                prior_fit = BBparams
            except Exception as e:
                print(f"WARNING: BB fit failed at epoch {i}: {e}")
                T_arr[i] = R_arr[i] = Terr_arr[i] = Rerr_arr[i] = covar_arr[i] = np.nan

    return T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr

def compute_bolometric_luminosity(T_arr, R_arr, Terr_arr, Rerr_arr, covar_arr):
    """
    Compute bolometric luminosity L = 4πR^2σT^4 with error propagation.
    """
    L = 4.0 * np.pi * R_arr**2 * sigsb * T_arr**4
    dL_dT = 16.0 * np.pi * R_arr**2 * sigsb * T_arr**3
    dL_dR = 8.0 * np.pi * R_arr * sigsb * T_arr**4
    var = (dL_dT * Terr_arr)**2 + (dL_dR * Rerr_arr)**2 + 2 * dL_dT * dL_dR * covar_arr
    L_err = np.sqrt(np.abs(var))
    return L, L_err


def compute_pseudo_bol(dense_lc, wvs, n_samples=500):
    """
    Compute pseudo-bolometric luminosity and its uncertainty by 
    Monte Carlo–propagating per-band mag errors through a λ-integral.

    Assumes:
      dense_lc[:,:,0] = −M_AB   (negative absolute AB magnitudes)
      dense_lc[:,:,1] = σ_M      (their 1σ errors)

    Parameters
    ----------
    dense_lc : ndarray, shape (T, F, 2)
      [:,:,0] = −M_AB, [:,:,1] = σ_M
    wvs : ndarray, shape (F,)
      Effective central wavelengths (Å)
    n_samples : int
      Number of MC samples to draw per epoch (default: 500)

    Returns
    -------
    bol_lum : ndarray, shape (T,)
      Mean pseudo-bolometric luminosity [erg/s]
    bol_err : ndarray, shape (T,)
      1σ uncertainty on pseudo-bolometric L [erg/s]
    """
    # constants
    c        = 2.99792458e10            # speed of light [cm/s]
    D10pc_cm = 10.0 * 3.085677581e18     # 10 pc in cm
    fourpiD2 = 4.0 * np.pi * D10pc_cm**2
    lam_cm   = wvs * 1e-8                # Å → cm, shape (F,)

    # unpack
    M_AB    = -dense_lc[:, :, 0]         # (T, F)
    sigma_M =  dense_lc[:, :, 1]         # (T, F)
    T, F    = M_AB.shape

    # pre-allocate
    L_samples = np.zeros((T, n_samples))

    # Monte Carlo loop
    for k in range(n_samples):
        # 1) perturb M_AB by its errors
        M_pert = M_AB + np.random.randn(T, F) * sigma_M
        
        # 2) mag → f_nu @10 pc [erg/s/cm²/Hz]
        fnu = 10.0 ** (-0.4 * (M_pert + 48.6))  # (T, F)

        # 3) f_nu → L_lambda @10 pc [erg/s/Å]
        #    Lλ = fν * (c / λ²) * 4π D²
        L_lambda = fnu * (c / lam_cm**2)[None, :] * fourpiD2  # (T, F)

        # 4) integrate in λ to get L [erg/s]
        L_samples[:, k] = np.trapz(L_lambda, x=lam_cm, axis=1)

    # compute mean and 1σ
    bol_lum = L_samples.mean(axis=1)
    bol_err = L_samples.std(axis=1)

    return bol_lum, bol_err