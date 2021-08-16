#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:06:32 2021

@author: vedantchandra
"""

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from . import utils
from . import models

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
    
    nwl, nfl, nivar = utils.cont_norm_lines(wl, fl, ivar,
                                            corvmodel.names,
                                            corvmodel.centres,
                                            corvmodel.windows,
                                            corvmodel.edges)
    
    _,nmodel = models.get_normalized_model(wl, corvmodel, params)
    resid = (nfl - nmodel) * np.sqrt(nivar)
    
    return resid

def xcorr_rv(wl, fl, ivar, corvmodel, params,
             min_rv = -1500, max_rv = 1500, 
             npoint = 250,
             quad_window = 300):
    """
    Find best RV via x-correlation on grid and quadratic fitting the peak.

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
    min_rv : float, optional
        lower end of RV grid. The default is -1500.
    max_rv : float, optional
        upper end of RV grid. The default is 1500.
    npoint : int, optional
        numver of equi-spaced points in RV grid. The default is 250.
    quad_window : float, optional
        window around minimum to fit quadratic model, 
        in km/s. The default is 300.

    Returns
    -------
    rv : float
        best-fit radial velocity.
    rvgrid : array_like
        grid of radial velocities.
    cc : array_like
        chi-square statistic evaluated at each RV.

    """
    
    if npoint is None:
        npoint = int(max_rv - min_rv)
        
    rvgrid = np.linspace(min_rv, max_rv, npoint)
    cc = np.zeros(npoint)
    params = corvmodel.make_params()
    
    residual = lambda params: normalized_residual(wl, fl, ivar, 
                                                  corvmodel, params)
    
    for ii,rv in enumerate(rvgrid):
        params['RV'].set(value = rv)
        chi = np.sum(residual(params)**2)
        cc[ii] = chi
        
    window = int(quad_window / np.diff(rvgrid)[0])
    
    argmin = np.argmin(cc)
    c1 = argmin - window
    c2 = argmin + window + 1

    rvgrid = rvgrid[c1:c2]
    cc = cc[c1:c2]

    pcoef = np.polyfit(rvgrid, cc, 2)

    rv = - 0.5 * pcoef[1] / pcoef[0]  
        
    return rv, rvgrid, cc

def fit_rv(wl, fl, ivar, corvmodel, params, fix_nonrv = True, 
           xcorr_kw = {}):
    """
    Use LMFIT to fit RV, after first estimating it by cross-correlation. 

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
    fix_nonrv : book, optional
        whether to fix all non-RV parameters. The default is True.
    xcorr_kw : dict, optional
        keywords to pass to xcorr_rv. The default is {}.

    Returns
    -------
    res : LMFIT MinimzerResult class
        rv-fitting results.
    rv_init : float
        initial guess RV from the x-correlation, for comparison purposes.

    """
    
    rv_init, rvgrid, cc = xcorr_rv(wl, fl, ivar, corvmodel, params,
                                   **xcorr_kw)
    
    if fix_nonrv:
        for param in params:
            params[param].set(vary = False)
        
    params['RV'].set(value = rv_init, vary = True)

    residual = lambda params: normalized_residual(wl, fl, ivar, 
                                                  corvmodel, params)
    
    res = lmfit.minimize(residual, params)
    
    return res, rv_init

def fit_corv(wl, fl, ivar, corvmodel, xcorr_kw = {},
                  iter_teff = False,
                  tpar = dict(tmin = 10000, tmax = 20000, nt = 2)):
    """
    Fit model parameters, x-corr RV, then LMFIT RV. 

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
    xcorr_kw : dict, optional
        keywords to pass to xcorr_rv. The default is {}.
    iter_teff : bool, optional
        whether to iterate over several initial teffs. The default is False.
    tpar : dict, optional
        initial teff iteration parameters. The default is 
        dict(tmin = 10000, tmax = 20000, nt = 2).

    Returns
    -------
    param_res : LMFIT MinimizerResult class
        result of fit to parameters, allowing everything to vary.
    rv_res : LMFIT MinimizerResult class
        result of RV fit from LMFIT.
    rv_init : float
        best RV from x-correlation, in km/s

    """
    
    params = corvmodel.make_params()
    
    residual = lambda params: normalized_residual(wl, fl, ivar, 
                                                  corvmodel, params)
    
    if iter_teff:
        minchi = 1e50
        init_teffs = np.linspace(tpar['tmin'], tpar['tmax'], tpar['nt'])
        for ii in range(tpar['nt']):
            params_i = params.copy()
            params_i['teff'].set(value = init_teffs[ii])
            resi = lmfit.minimize(residual, params_i)
            
            if resi.redchi < minchi:
                param_res = resi
                minchi = resi.redchi
            else:
                continue
    else:
        param_res = lmfit.minimize(residual, params)
        
    bestparams = param_res.params.copy()
    
    rv_res, rv_init = fit_rv(wl, fl, ivar, corvmodel, bestparams, **xcorr_kw)
            
    return param_res, rv_res, rv_init