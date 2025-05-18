import numpy as np
import scipy.optimize as opt
from george import GP, kernels
from george.modeling import Model
import importlib_resources
from scipy import interpolate as interp

epsilon = 0.001  # small wavelength tolerance

def chi_square(dat, model, uncertainty):
    chi2 = 0.0
    for i in range(len(dat)):
        chi2 += ((model[i] - dat[i]) / uncertainty[i]) ** 2
    return chi2

def generate_template(filter_wv, sn_type):
    pkg = importlib_resources.files("extrabol.template_bank")
    fn  = pkg / f"smoothed_sn{sn_type}.npz"
    data = np.load(fn)

    times = data["time"]
    wavs  = data["wavelength"]
    fl    = data["f_lambda"]

    # 1) overlap wavelengths
    mw = (wavs >= filter_wv.min()) & (wavs <= filter_wv.max())
    times, wavs, fl = times[mw], wavs[mw], fl[mw]

    # 2) every other time
    m2 = (times % 2 == 0)
    times, wavs, fl = times[m2], wavs[m2], fl[m2]

    # 3) every 20 Å
    m3 = (wavs % 20 == 0)
    times, wavs, fl = times[m3], wavs[m3], fl[m3]

    # 4) drop t<1
    m4 = (times >= 1.0)
    times, wavs, fl = times[m4], wavs[m4], fl[m4]

    # 5) shift peak to zero
    peak = np.argmax(fl)
    times = times - times[peak]

    t_u = np.unique(times)
    w_u = np.unique(wavs)
    grid = np.full((t_u.size, w_u.size), np.nan)

    for i, tt in enumerate(t_u):
        for j, ww in enumerate(w_u):
            idx = np.where((times == tt) & (wavs == ww))[0]
            if idx.size:
                grid[i, j] = fl[idx].mean()

    # log‐flux conversion
    for j, lam in enumerate(w_u):
        grid[:, j] = 2.5 * np.log10((lam**2) * grid[:, j])

    return interp.RectBivariateSpline(t_u, w_u, grid)


class SNModel(Model):
    """
    GP mean function wrapping the SN template, with no free parameters.
    """
    def __init__(self, tpl, f_s, t_s, t_t, flux_corr):
        self.tpl = tpl
        self.f_s = f_s
        self.t_s = t_s
        self.t_t = t_t
        self.flux_corr = flux_corr

    @property
    def parameter_vector(self):
        return np.zeros(0)

    @parameter_vector.setter
    def parameter_vector(self, v):
        pass

    @property
    def unfrozen_mask(self):
        return np.zeros(0, dtype=bool)

    def get_value(self, X):
        t_vals = (X[:, 0] / self.t_t) + self.t_s
        w_vals = X[:, 1]
        return np.array([
            self.tpl(tt, ww)[0] + self.f_s - self.flux_corr
            for tt, ww in zip(t_vals, w_vals)
        ])


def fit_template(wv, tpl, filts, wv_corr,
                 flux, time, errs, z,
                 output_chi=False, output_params=True):
    A_opts, t_c_opts, t_s_opts, chis = [], [], [], []
    for wavelength in wv:
        def model(t, filt, A, t_c, t_s):
            t_corr = (np.sort(t) / t_s) + t_c
            return tpl(t_corr, filt).flatten() + A

        def fitfunc(t, A, t_c, t_s):
            return model(t, wavelength, A, t_c, t_s)

        mask = np.isclose(filts * 1000 + wv_corr, wavelength)
        popt, _ = opt.curve_fit(
            fitfunc, time[mask], flux[mask],
            p0=[20, 0, 1 + z],
            maxfev=8000,
            bounds=([-np.inf, -np.inf, 0], np.inf)
        )
        A_opts.append(popt[0])
        t_c_opts.append(popt[1])
        t_s_opts.append(popt[2])

        total = 0.0
        for filt in wv:
            pred = model(time[mask], filt, *popt)
            total += chi_square(flux[mask], pred, errs[mask])
        chis.append(total)

    best = int(np.argmin(chis))
    if output_chi and output_params:
        return A_opts[best], t_c_opts[best], t_s_opts[best], chis[best]
    if output_chi:
        return chis[best]
    if output_params:
        return A_opts[best], t_c_opts[best], t_s_opts[best]
    return 0


def test(lc, wv_corr, z):
    lc = lc.T
    time, flux, filts, errs = lc[:,0], lc[:,1], lc[:,2], lc[:,3]
    ufilts = np.unique(filts)
    u_ang   = ufilts * 1000 + wv_corr

    templates = ['1a','1bc','2p','2l']
    chis = []
    for tpl in templates:
        fn = generate_template(u_ang, tpl)
        chis.append(fit_template(
            u_ang, fn, filts, wv_corr, flux, time, errs,
            z, output_chi=True, output_params=False
        ))
    return templates[int(np.argmin(chis))]


def interpolate(lc, wv_corr, flux_corr, sn_type, use_mean,
                z, verbose, stepsize, kernel_width=None):
    """
    Interpolate the LC using a 2D Gaussian Process (GP),
    optionally with a SN template mean.
    """
    times, fluxes, wv_effs, errs, _ = lc
    stacked = np.vstack([times, wv_effs]).T
    ufilts = np.unique(wv_effs)
    u_ang   = ufilts * 1000 + wv_corr
    nfilts = len(ufilts)

    tmin, tmax = int(np.floor(times.min())), int(np.ceil(times.max()))
    dense_times = np.arange(tmin, tmax + 1, stepsize)

    if use_mean:
        tpl = generate_template(u_ang, sn_type)
        if verbose:
            print("Fitting template…")
        f_s, t_s, t_t = fit_template(
            u_ang, tpl, wv_effs, wv_corr,
            fluxes, times, errs, z
        )
        mean_func = SNModel(tpl, f_s, t_s, t_t, flux_corr)
    else:
        mean_func = 0

    kernel = np.var(fluxes) * kernels.Matern32Kernel(
        kernel_width or [12.0, 0.1], ndim=2
    )
    gp = GP(kernel, mean=mean_func)

    # filter any non‐finite uncertainties
    mask = np.isfinite(errs)
    x_train, y_err = stacked[mask], errs[mask]
    gp.compute(x_train, y_err)

    if kernel_width is None:
        def nll(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(fluxes)
        def grad(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(fluxes)

        res = opt.minimize(
            nll,
            gp.get_parameter_vector(),
            jac=grad,
            bounds=[(None,None)]*len(gp.get_parameter_vector())
        )
        gp.set_parameter_vector(res.x)

    # build prediction grid
    N = len(dense_times) * nfilts
    x_pred = np.zeros((N,2))
    for i, tt in enumerate(dense_times):
        x_pred[i*nfilts:(i+1)*nfilts,0] = tt
        x_pred[i*nfilts:(i+1)*nfilts,1] = ufilts

    pred, var = gp.predict(fluxes, x_pred, return_var=True)

    dense_f = np.zeros((len(dense_times), nfilts))
    dense_e = np.zeros_like(dense_f)
    for j in range(nfilts):
        sel = np.isclose(x_pred[:,1], ufilts[j])
        dense_f[:,j] = pred[sel]
        dense_e[:,j] = np.sqrt(var[sel])

    dense_lc = np.dstack((dense_f, dense_e))

    test_y = []
    test_t = dense_times.copy()

    if use_mean:
        test_y = []
        for w in u_ang:
            # call the spline with a vector of times and a 1-element wavelength array
            arr = tpl(dense_times, np.array([w]))  # shape (len(dense_times), 1)
            # take the first (and only) column, then add your stretch offset f_s
            test_y.append(arr[:, 0] + f_s)
        test_y = np.array(test_y)  # now shape (nfilts, len(dense_times))
    else:
        test_y = np.empty((0, len(dense_times)))

    return dense_lc, test_y, test_t, dense_times
