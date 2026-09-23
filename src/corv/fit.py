#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:06:32 2021

@author: vedantchandra

Fitting proceeds in two stages:
    1. fit_params: least-squares fit of teff and logg (RV free as a nuisance
       parameter).
    2. fit_rv: with teff and logg fixed, scan chi-square over a grid of RVs and
       fit a parabola to the minimum to get RV and its uncertainty.
fit_corv runs both.
"""

import warnings

import lmfit
import numpy as np

from . import utils
from . import models

def _normalize_data(wl, fl, ivar, corvmodel):
    """Continuum-normalize and crop the data to the lines of corvmodel."""
    _, nfl, nivar = utils.cont_norm_lines(wl, fl, ivar,
                                          corvmodel.names,
                                          corvmodel.centres,
                                          corvmodel.windows,
                                          corvmodel.edges)
    return nfl, nivar

def _residual(wl, nfl, nivar, corvmodel, params):
    """
    Error-scaled residuals against already-normalized data. Masked pixels
    (nivar = 0, which cont_norm_line pairs with NaN flux) contribute zero.
    """
    _, nmodel = models.get_normalized_model(wl, corvmodel, params)
    with np.errstate(invalid = 'ignore'):
        return np.where(nivar > 0, (nfl - nmodel) * np.sqrt(nivar), 0.)

def normalized_residual(wl, fl, ivar, corvmodel, params):
    """
    Error-scaled residuals between data and evaluated model

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class
        parameters at which to evaluate corvmodel.

    Returns
    -------
    resid : array_like
        error-scaled residual array.

    """
    nfl, nivar = _normalize_data(wl, fl, ivar, corvmodel)
    return _residual(wl, nfl, nivar, corvmodel, params)

def fit_params(wl, fl, ivar, corvmodel, params = None, init_teffs = (12000,)):
    """
    Least-squares fit of the model parameters (teff, logg, and RV as a
    nuisance parameter), started from each of init_teffs.

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class, optional
        starting parameters. The default is corvmodel.make_params().
    init_teffs : array_like, optional
        starting teffs to try. The default is (12000,). Ignored for models
        without a teff parameter.

    Returns
    -------
    param_res : LMFIT MinimizerResult class
        the fit with the lowest reduced chi-square.

    """
    if params is None:
        params = corvmodel.make_params()
    nfl, nivar = _normalize_data(wl, fl, ivar, corvmodel)
    residual = lambda p: _residual(wl, nfl, nivar, corvmodel, p)

    if 'teff' not in params:
        init_teffs = [None] # e.g. make_balmer_model

    param_res = None
    for teff in init_teffs:
        params_i = params.copy()
        if teff is not None:
            params_i['teff'].set(value = teff)
        res = lmfit.minimize(residual, params_i)
        if param_res is None or res.redchi < param_res.redchi:
            param_res = res
    return param_res

def fit_rv(wl, fl, ivar, corvmodel, params,
           min_rv = -1500, max_rv = 1500,
           npoints = 500,
           quad_window = 300,
           plot = False, path = None):
    """
    Find the best RV by chi-square minimization on an RV grid, with all other
    parameters fixed, then fit a parabola to the minimum.

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    params : LMFIT Parameters class
        parameters at which to evaluate corvmodel. Not modified.
    min_rv : float, optional
        lower end of RV grid. The default is -1500.
    max_rv : float, optional
        upper end of RV grid. The default is 1500.
    npoints : int, optional
        number of points in the RV grid. The default is 500.
    quad_window : float, optional
        half-width of the window around the minimum used to fit the
        parabola, in km/s. The default is 300.
    plot : bool, optional
        whether to plot the chi-square curve. The default is False.
    path : str, optional
        where to save the plot. The default is None (show it).

    Returns
    -------
    rv : float
        best-fit radial velocity.
    e_rv : float
        1-sigma uncertainty from delta chi-square = 1. NaN if the parabola
        fit fails.
    redchi : float
        reduced chi-square at the best-fit RV.
    rvgrid : array_like
        RV grid within the fitting window.
    chi2 : array_like
        chi-square evaluated at each RV in rvgrid.

    """
    params = params.copy()
    nfl, nivar = _normalize_data(wl, fl, ivar, corvmodel)
    dof = np.sum(nivar > 0) - 1

    rvgrid = np.linspace(min_rv, max_rv, npoints)
    chi2 = np.zeros(len(rvgrid))
    for ii, rv in enumerate(rvgrid):
        params['RV'].set(value = rv)
        chi2[ii] = np.nansum(_residual(wl, nfl, nivar, corvmodel, params)**2)

    window = max(int(quad_window / np.diff(rvgrid)[0]), 1)
    argmin = np.nanargmin(chi2)
    sel = slice(max(argmin - window, 0), argmin + window + 1)
    rvgrid, chi2 = rvgrid[sel], chi2[sel]

    pcoef = np.polyfit(rvgrid, chi2, 2)
    if pcoef[0] > 0:
        rv = -0.5 * pcoef[1] / pcoef[0]
        e_rv = 1 / np.sqrt(pcoef[0])
        chi2_min = np.interp(rv, rvgrid, chi2)
    else:
        warnings.warn('chi-square curve has no minimum; returning the grid '
                      'minimum with e_rv = nan')
        rv = rvgrid[np.nanargmin(chi2)]
        e_rv = np.nan
        chi2_min = np.nanmin(chi2)
    redchi = chi2_min / dof if dof > 0 else np.nan

    if plot:
        utils.plot_chi2(rvgrid, chi2, rv, e_rv, pcoef, path = path)

    return rv, e_rv, redchi, rvgrid, chi2

xcorr_rv = fit_rv # backwards-compatible name

def fit_corv(wl, fl, ivar, corvmodel, init_teffs = (12000,), rv_kw = None,
             xcorr_kw = None):
    """
    Fit teff and logg with fit_params, then RV with fit_rv.

    Parameters
    ----------
    wl : array_like
        wavelengths in Angstroms.
    fl : array_like
        flux array.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT model with normalization instructions.
    init_teffs : array_like, optional
        starting teffs for fit_params. The default is (12000,).
    rv_kw : dict, optional
        keywords to pass to fit_rv. The default is None.
    xcorr_kw : dict, optional
        old name for rv_kw.

    Returns
    -------
    rv : float
        best-fit radial velocity in km/s.
    e_rv : float
        uncertainty on rv in km/s.
    redchi : float
        reduced chi-square at rv.
    param_res : LMFIT MinimizerResult class
        result of fit_params, with the RV parameter set to rv and e_rv.

    """
    rv_kw = rv_kw or xcorr_kw or {}

    param_res = fit_params(wl, fl, ivar, corvmodel, init_teffs = init_teffs)
    rv, e_rv, redchi, _, _ = fit_rv(wl, fl, ivar, corvmodel, param_res.params, **rv_kw)

    param_res.params['RV'].value = rv
    param_res.params['RV'].stderr = e_rv

    return rv, e_rv, redchi, param_res
