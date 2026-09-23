#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:08:00 2021

@author: vedantchandra

Helper functions: continuum normalization, spectrum utilities, wavelength
conversions, and plotting. matplotlib is only imported by the plotting
functions.
"""

import warnings

import numpy as np
from bisect import bisect_left
import scipy.ndimage

c_kms = 2.99792458e5 # speed of light in km/s

### CONTINUUM NORMALIZATION ###

def cont_norm_line(wl, fl, ivar, centre, window, edge):
    """
    Continuum-normalizes a single absorption/emission line.

    Parameters
    ----------
    wl : array_like
        wavelength.
    fl : array_like
        flux.
    ivar : array_like
        inverse-variance.
    centre : float
        line centroid.
    window : float
        selected region on either side of line, in Angstrom.
    edge : int
        number of pixels on each edge of the region used to fit a linear
        continuum. Pixels with ivar <= 0 are excluded from the fit.

    Returns
    -------
    wl : array_like
        cropped wavelength array.
    norm_fl : array_like
        cropped and normalized flux array.
    norm_ivar : array_like
        cropped and normalized inverse-variance array.

    """
    c1 = bisect_left(wl, centre - window)
    c2 = bisect_left(wl, centre + window)
    wl, fl, ivar = wl[c1:c2], fl[c1:c2], ivar[c1:c2]

    if edge < 1:
        raise ValueError('edge must be at least 1 pixel')
    if len(wl) <= 2 * edge:
        raise ValueError('line at %.1f AA has %i pixels in its window, too few for '
                         'edge = %i' % (centre, len(wl), edge))

    mask = np.zeros(len(wl), dtype = bool)
    mask[:edge] = True
    mask[-edge:] = True
    mask &= (ivar > 0)

    if mask.sum() < 2:
        warnings.warn('line at %.1f AA has no usable continuum pixels; '
                      'masking the line' % centre)
        return wl, np.full(len(wl), np.nan), np.zeros(len(wl))

    p = np.polynomial.polynomial.polyfit(wl[mask], fl[mask], 1)
    continuum = np.polynomial.polynomial.polyval(wl, p)
    norm_fl = fl / continuum
    norm_ivar = ivar * continuum**2
    return wl, norm_fl, norm_ivar

def cont_norm_lines(wl, fl, ivar, names, centres, windows, edges):
    """
    Runs cont_norm_line on each line in names and concatenates the results.

    Parameters
    ----------
    wl, fl, ivar : array_like
        wavelength, flux and inverse-variance.
    names : list
        line keys, in the order they should be concatenated.
    centres, windows, edges : dict
        per-line arguments to cont_norm_line, keyed by name.

    Returns
    -------
    nwl, nfl, nivar : array_like
        concatenated cropped wavelengths, normalized flux and inverse-variance.

    """
    lines = [cont_norm_line(wl, fl, ivar, centres[line], windows[line], edges[line])
             for line in names]
    return tuple(np.concatenate([l[k] for l in lines] or [[]]) for k in range(3))

def continuum_normalize(wl, fl, ivar = None, avg_size = 300, ret_cont = False):
    """
    Normalizes a whole spectrum by a running median of width avg_size (AA).

    Returns (wl, fl_norm), plus ivar_norm if ivar is given, plus the
    continuum if ret_cont is True.
    """
    fl_cont = np.zeros(np.size(fl))
    for i in range(np.size(wl)):
        wl_clip = ((wl[i] - avg_size/2) < wl) & (wl < (wl[i] + avg_size/2))
        fl_cont[i] = np.median(fl[wl_clip])

    out = [wl, fl / fl_cont]
    if ivar is not None:
        out.append(ivar * fl_cont**2)
    if ret_cont:
        out.append(fl_cont)
    return tuple(out)

### SPECTRUM UTILITIES ###

def crrej(wl, fl, ivar, nsig = 3, medwindow = 11, plot = False):
    """
    Masks cosmic rays: pixels more than nsig sigma from a running median of
    width medwindow pixels. Masked pixels get ivar = 0 and linearly
    interpolated flux. The inputs are not modified.

    Returns
    -------
    wl, corr_fl, corr_ivar : array_like
    """
    medfl = scipy.ndimage.median_filter(fl, medwindow)
    zscore = (fl - medfl) * np.sqrt(ivar)
    crmask = (np.abs(zscore) > nsig) | (ivar == 0)

    corr_ivar = ivar.copy()
    corr_ivar[crmask] = 0
    corr_fl = np.interp(wl, wl[~crmask], fl[~crmask])

    if plot:
        plot_crrej(wl, fl, medfl, zscore)
        print('%i pixels rejected' % np.sum(crmask))

    return wl, corr_fl, corr_ivar

def doppler_shift(wl, fl, dv):
    """Flux of the spectrum (wl, fl) Doppler-shifted by dv km/s, sampled at wl."""
    df = np.sqrt((1 - dv/c_kms)/(1 + dv/c_kms))
    return np.interp(wl * df, wl, fl)

def get_medsn(wl, fl, ivar):
    """
    Signal-to-noise in the 5400-5800 AA continuum.

    Returns
    -------
    medsn : float
        median of fl * sqrt(ivar).
    sn_est : float
        1 / scatter about a quadratic continuum fit, independent of ivar.
    """
    wlsel = (wl > 5400) & (wl < 5800)
    cwl, cfl, civar = wl[wlsel], fl[wlsel], ivar[wlsel]
    medsn = np.nanmedian(cfl * np.sqrt(civar))
    contnorm = cfl / np.polyval(np.polyfit(cwl, cfl, 2), cwl)
    return medsn, 1 / np.std(contnorm)

def da_nlte_1d_correction(teff, logg):
    """Returns (teff_shift, logg_shift) from the fitted correction functions."""
    A = np.array([1.0947335e-03, -1.8716231e-01, 1.9350009e-02, 6.4821613e-01,
                  -2.2863187e-01, 5.8699232e-01, -1.0729871e-01, 1.1009070e-01])
    B = np.array([7.5209868E-04, -9.2086619E-01, 3.1253746E-01, -1.0348176E+01,
                  6.5854716E-01, 4.2849862E-01, -8.8982873E-02, 1.0199718E+01,
                  4.9277883E-02, -8.6543477E-01, 3.6232756E-03, -5.8729354E-02])
    teff0 = (teff - 10000) / 1000
    logg0 = (logg - 8) / 1
    teff_shift =(A[0]+(A[1]+A[6]*teff0+A[7]*logg0)*np.exp(-(A[2]+A[4]*teff0+A[5]*logg0)**2*((teff0-A[3])**2))) * 1000
    logg_shift = (B[0]+B[4]*np.exp(-B[5]*((teff0-B[6])**2)))+B[1]*np.exp(-B[2]*((teff0-(B[3]+B[7]*np.exp(-(B[8]+B[10]*teff0+B[11]*logg0)**2*((teff0-B[9])**2))))**2))
    return teff_shift, logg_shift

### WAVELENGTH CONVERSIONS ###

def air2vac(wv):
    """
    Air to vacuum wavelengths, formula from Morton 1991 ApJS, 77, 119.

    Parameters
    ----------
    wv : array_like
        air wavelengths in Angstrom.

    Returns
    -------
    array_like
        vacuum wavelengths in Angstrom.

    """
    _tl = 1.e4/np.array(wv)
    return (np.array(wv) * (1. + 6.4328e-5 + 2.94981e-2
                            / (146. - _tl**2) + 2.5540e-4 / (41. - _tl**2)))

def vac2air(wv):
    """
    Vacuum to air wavelengths, formula from Morton 1991 ApJS, 77, 119.

    Parameters
    ----------
    wv : array_like
        vacuum wavelengths in Angstrom.

    Returns
    -------
    array_like
        air wavelengths in Angstrom.

    """
    _tl = 1.e4/np.array(wv)
    return (np.array(wv) / (1. + 6.4328e-5 + 2.94981e-2
                            / (146. - _tl**2) + 2.5540e-4 / (41. - _tl**2)))

### PLOTTING ###

def lineplot(wl, fl, ivar, corvmodel, params, gap = 0.3, printparams = True,
             figsize = (10, 7)):
    """
    Plots each normalized line of the data (black) and model (red), offset
    vertically by gap.

    Parameters
    ----------
    wl, fl, ivar : array_like
        wavelength, flux and inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class
        parameters at which to evaluate corvmodel.
    gap : float, optional
        vertical offset between lines. The default is 0.3.
    printparams : bool, optional
        annotate teff, logg (when the model has them) and reduced
        chi-square. The default is True.
    figsize : tuple, optional
        figure size. The default is (10, 7).

    Returns
    -------
    f : matplotlib Figure

    """
    import matplotlib.pyplot as plt

    model = corvmodel.eval(params, x = wl)
    chi2 = 0
    dof = 0

    f = plt.figure(figsize = figsize)
    for ii, line in enumerate(corvmodel.names):
        args = (corvmodel.centres[line], corvmodel.windows[line], corvmodel.edges[line])
        cwl, cfl, civar = cont_norm_line(wl, fl, ivar, *args)
        _, cmodel, _ = cont_norm_line(wl, model, model, *args)

        dlam = cwl - corvmodel.centres[line]
        plt.plot(dlam, cfl - ii * gap, 'k')
        plt.plot(dlam, cmodel - ii * gap, 'r')

        chi2 += np.nansum((cfl - cmodel)**2 * civar)
        dof += np.sum(civar > 0)

    plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
    plt.ylabel('Normalized Flux')

    if printparams:
        redchi = chi2 / (dof - sum(p.vary for p in params.values()))
        stderr = lambda name: np.nan if params[name].stderr is None else params[name].stderr
        labels = [r'$\chi_r^2$ = %.2f' % redchi]
        if 'logg' in params:
            labels.append(r'$\log{g} = %.2f \pm %.2f $' % (params['logg'].value, stderr('logg')))
        if 'teff' in params:
            labels.append(r'$T_{\mathrm{eff}} = %.0f \pm %.0f\ K$' %
                          (params['teff'].value, stderr('teff')))
        for jj, label in enumerate(labels[::-1]):
            plt.text(0.97, 0.05 + 0.07 * jj, label, transform = plt.gca().transAxes,
                     fontsize = 14, ha = 'right')

    plt.ylim(-gap * ii, 1 + gap)
    return f

def plot_chi2(rvgrid, chi2, rv, e_rv, pcoef, path = None):
    """
    Plots a chi-square curve from corv.fit.fit_rv with its parabola fit.
    Saves to path if given, otherwise shows the figure.
    """
    import matplotlib.pyplot as plt

    xgrid = np.linspace(min(rvgrid), max(rvgrid), 50)
    plt.figure(figsize = (10,5))
    plt.plot(rvgrid, chi2, label = r'Actual $\chi^2$ curve')
    plt.plot(xgrid, np.polyval(pcoef, xgrid), label = r'Fitted $\chi^2$ curve')
    plt.axvline(x = rv)
    plt.axvline(x = rv + e_rv, ls = ':')
    plt.axvline(x = rv - e_rv, ls = ':')
    plt.axhline(y = np.polyval(pcoef, rv), label = r'Minimum $\chi^2$')
    plt.legend()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_crrej(wl, fl, medfl, zscore):
    """Diagnostic plots for crrej: flux vs. running median, then |z-score|."""
    import matplotlib.pyplot as plt

    plt.plot(wl, fl)
    plt.plot(wl, medfl)
    plt.show()

    plt.title('crrej z-score')
    plt.plot(wl, np.abs(zscore))
    plt.ylim(-0.5, 5)
    plt.show()
