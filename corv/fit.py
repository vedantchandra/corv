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