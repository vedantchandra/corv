#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:04:32 2021

@author: Vedant Chandra, Keith P. Inight

Notes:
    - A 'corvmodel' is an LMFIT-like class that contains extra information.
    - The continuum-normalization method is linear, line-by-line.
    - Normalization and cropping happens AFTER the template is computed and 
    doppler-shifted. 
    - Current plan for fitting: corvmodel is used to evaluate model (with RV)
    and define parameters. A separate residual function uses this, and also
    defines the type of continuum-normalization. Then that residual function
    can be minimized in a variety of ways; e.g. leastsq for template-fitting,
    xcorr over RV for individual exposures. 
    
To-do:
    - Add convolution parameter to bring models to instrument resolution
    - Perhaps even convolve the LSF in a wavelength-dependent manner
    - Add Koester DB models, 
"""

import numpy as np
from lmfit.models import Model, ConstantModel, VoigtModel
import pickle
import os
basepath = os.path.dirname(os.path.abspath(__file__))
print(basepath)

from . import utils

c_kms = 2.99792458e5 # speed of light in km/s

# add epsilon?
default_centres = dict(a = 6564.61, b = 4862.68, g = 4341.68, d = 4102.89)
default_windows = dict(a = 200, b = 200, g = 80, d = 80)
default_edges = dict(a = 50, b = 50, g = 20, d = 20)
default_names = ['d', 'g', 'b', 'a']

# CONTINUUM NORMALIZATION

def normalize_lines(wl, fl, ivar, corvmodel):
    """
    Normalize all absorption lines defined by a given model type.

    Parameters
    ----------
    wl : array_like
        wavelength in AA.
    fl : array_like
        flux.
    ivar : array_like
        inverse-variance.
    corvmodel : LMFIT Model class
        LMFIT Model with centres and lines added.

    Returns
    -------
    tuple
        tuple of normalized wavelength, flux and inverse-variance arrays.

    """
    nwl = [];
    nfl = [];
    nivar = [];
    for line in corvmodel.names:

        nwli, nfli, nivari = utils.cont_norm_line(wl, fl, ivar, 
                                                  corvmodel.centres[line], 
                                                  corvmodel.windows[line],
                                                  corvmodel.edges[line])
        nwl.extend(nwli)
        nfl.extend(nfli)
        nivar.extend(nivari)
        
    return np.array(nwl), np.array(nfl), np.array(nivar)

def get_normalized_model(wl, corvmodel, params):
    """
    Evaluates and continuum-normalizes a given corvmodel. 

    Parameters
    ----------
    wl : array_like
        wavelength in Angstrom.
    corvmodel : LMFIT model class
        model class with line attributes defined.
    params : LMFIT Parameters class
        parameters at which to evaluate model.

    Returns
    -------
    nwl : array_like
        cropped wavelengths in Angstrom.
    nfl : TYPE
        cropped and continuum-normalized flux.

    """
    flux = corvmodel.eval(params, x = wl)
    
    # pass a dummy ivar value below
    nwl, nfl, _ = normalize_lines(wl, flux, flux, corvmodel)
    
    return nwl, nfl

# MODEL DEFINITIONS
# Balmer Model


def make_balmer_model(nvoigt=1, 
                 centres = default_centres, 
                 windows = default_windows, 
                 edges = default_edges,
                 names = default_names):
    """
    Models each Balmer line as a (sum of) Voigt profiles

    Parameters
    ----------
    nvoigt : int, optional
        number of Voigt profiles per line. The default is 1.
    centres : dict, optional
        rest-frame line centres. The default is default_centres.
    windows : dict, optional
        region around each line in pixels. The default is default_windows.
    edges : TYPE, optional
        edge regions used to fit continuum. The default is default_edges.
    names : TYPE, optional
        line keys in ascending order of lambda. The default is default_names.

    Returns
    -------
    model : LMFIT model
        LMFIT-style model that can be evaluated and fitted.

    """

    model = ConstantModel()

    for line in names:
        for n in range(nvoigt):
            model -= VoigtModel(prefix = line + str(n) + '_')
   
    model.set_param_hint('c', value = 1)
  
    model.set_param_hint('RV', value = 0, min = -2500, max = 2500)
  
    for name in names:
        for n in range(nvoigt):
            pref = name + str(n)
            model.set_param_hint(pref + '_sigma', value = 15, min = 0)
            model.set_param_hint(pref + '_amplitude', value = 15, min = 0)
            if n == 0:
                restwl = str(centres[name])
                model.set_param_hint(pref + '_center', 
                                     expr = restwl + ('/ '
                                                      'sqrt((1 - '
                                                      'RV/2.99792458e5)/'
                                                      '(1 + '
                                                      'RV/2.99792458e5))'))
            elif n > 0:
                model.set_param_hint(pref + '_center', 
                                     expr = name + '0_center', vary = False)
    model.centres = centres
    model.windows = windows
    model.names = names
    model.edges = edges

    return model

# Koester DA Model

wd_interp = pickle.load(open(basepath + '/pkl/koester_interp.pkl', 'rb'))

def get_koester(x, teff, logg, RV):
    """
    Interpolates Koester (2010) DA models

    Parameters
    ----------
    x : array_like
        wavelength in Angstrom.
    teff : float
        effective temperature in K.
    logg : float
        log surface gravity in cgs.

    Returns
    -------
    flam : array_like
        synthetic flux interpolated at the requested parameters.

    """
    df = np.sqrt((1 - RV/c_kms)/(1 + RV/c_kms))
    x_shifted = x * df
    flam = 10**wd_interp((logg, np.log10(teff), np.log10(x_shifted)))
    return flam


def make_koester_model(centres = default_centres, 
                       windows = default_windows, 
                       edges = default_edges,
                       names = default_names):
    """
    

    Parameters
    ----------
    centres : dict, optional
        rest-frame line centres. The default is default_centres.
    windows : dict, optional
        region around each line in pixels. The default is default_windows.
    edges : TYPE, optional
        edge regions used to fit continuum. The default is default_edges.
    names : TYPE, optional
        line keys in ascending order of lambda. The default is default_names.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    
    model = Model(get_koester,
                  independent_vars = ['x'],
                  param_names = ['teff', 'logg', 'RV'])
    
    model.set_param_hint('teff', min = 3251, max = 39999, value = 12000)
    model.set_param_hint('logg', min = 6.01, max = 9.49, value = 8)
    model.set_param_hint('RV', min = -2500, max = 2500, value = 0)
    
    
    model.centres = centres
    model.windows = windows
    model.names = names
    model.edges = edges
    
    return model