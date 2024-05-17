#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:04:32 2021

author: Vedant Chandra, Keith P. Inight

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
import scipy 

from . import utils

basepath = os.path.dirname(os.path.abspath(__file__))
modpath = basepath[:-8] + 'models/'
#wd_interp = pickle.load(open('/home/arseneau/research/white-dwarfs/models/koester_interp_da.pkl', 'rb'))

base_wavl_da, warwick_da_interp,warwick_da_interp_low_logg, warwick_da_table = utils.build_warwick_da()

def fetch_montreal_da_table():
    return montreal_da_table


from . import utils

c_kms = 2.99792458e5 # speed of light in km/s

# add epsilon?
default_centres =  dict(a = 6564.61, b = 4862.68, g = 4341.68, d = 4102.89,
                 e = 3971.20, z = 3890.12, n = 3835.5,
             t = 3799.5)
default_windows = dict(a = 100, b = 100, g = 85, d = 70, e = 30,
                  z = 25, n = 15, t = 10)
default_edges = dict(a = 25, b = 25, g = 20, d = 20, 
                e = 5, z = 5, n = 5, t = 4)

default_names = ['n', 'z', 'e', 'd', 'g', 'b', 'a']
### MODEL DEFINITIONS ###

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

if modpath!='no path selected':
    try:
        wd_interp = pickle.load(open(modpath, 'rb'))
    except:
        print('We could not find the pickled WD models')
        inpmod3=input('Please enter a new path to the Koester models.')
        modpathnew=inpmod3
        try:
            wd_interp = pickle.load(open(modpathnew, 'rb'))
        except:
            print('We could not find the pickled WD models. If you need to use these models, please re-import corv with the proper path.')


def get_koester(x, teff, logg, RV, res):
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

    flam = np.zeros_like(x_shifted) * np.nan

    in_bounds = (x_shifted > 3600) & (x_shifted < 9000)
    flam[in_bounds] = 10**wd_interp((logg, np.log10(teff), np.log10(x_shifted[in_bounds])))

    flam = flam / np.nanmedian(flam) # bring to order unity
    
    dx = np.median(np.diff(x))
    window = res / dx
    
    flam = scipy.ndimage.gaussian_filter1d(flam, window)
    
    return flam


def make_koester_model(resolution = 1, centres = default_centres, 
                       windows = default_windows, 
                       edges = default_edges,
                       names = default_names):
    """
    

    Parameters
    ----------
    resolution : float, optional
        gaussian sigma in AA by which the models are convolved. 
        The default is 1.
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
                  param_names = ['teff', 'logg', 'RV', 'res'])
    
    model.set_param_hint('teff', min = 3001, max = 39999, value = 12000)
    model.set_param_hint('logg', min = 4.51, max = 9.49, value = 8)
    model.set_param_hint('RV', min = -2500, max = 2500, value = 0)
    model.set_param_hint('res', value = resolution, min = 0, vary = False)
    
    
    model.centres = centres
    model.windows = windows
    model.names = names
    model.edges = edges
    
    return model

# Montreal DA Model

#try:
#montreal_da_interp = pickle.load(open(basepath + '/models/montreal_da.pkl', 'rb'))
#base_wavl_da = np.load(basepath + '/models/montreal_da_wavl.npy')
#except:
#    print('could not find pickled Montreal DA WD models')

def get_warwick_da(x, teff, logg, RV, res):
    """
    Interpolates Montreal DA models for logg>7 dex

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

    flam = np.zeros_like(x_shifted) * np.nan

    in_bounds = (x_shifted > 3600) & (x_shifted < 9000)
    flam[in_bounds] = np.interp(x_shifted[in_bounds], base_wavl_da, warwick_da_interp((teff, logg)))

    #NICOLE BUG FIX
    norm=np.nanmedian(flam)
    flam = flam / norm # bring to order unity

    #NICOLE BUG FIX
    if norm==0:
        print("Median flux is 0. The fit has moved outside of the valid regime of the Montreal Models. These models cannot handle WDs with logg<7.5 and Teff<5000 K or >14000K")
        print("Tried Teff= "+str(teff)+" K, Logg="+str(logg)+" dex")
        
    dx = np.median(np.diff(x))
    window = res / dx
    
    flam = scipy.ndimage.gaussian_filter1d(flam, window)
    
    return flam


def make_warwick_da_model(resolution = 1, centres = default_centres, 
                       windows = default_windows, 
                       edges = default_edges,
                       names = default_names):
    """
    For logg>7 dex

    Parameters
    ----------
    resolution : float, optional
        gaussian sigma in AA by which the models are convolved. 
        The default is 1.
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
    
    model = Model(get_warwick_da,
                  independent_vars = ['x'],
                  param_names = ['teff', 'logg', 'RV', 'res'])
    
    model.set_param_hint('teff', min = 4001, max = 35000, value = 12000)
    model.set_param_hint('logg', min = 7, max = 9, value = 8)
    model.set_param_hint('RV', min = -2500, max = 2500, value = 0)
    model.set_param_hint('res', value = resolution, min = 0, vary = False)
    
    
    model.centres = centres
    model.windows = windows
    model.names = names
    model.edges = edges
    
    return model

#NICOLE BUG FIX
def get_warwick_da_low_logg(x, teff, logg, RV, res):
    """
    Interpolates Montreal DA models for loggs below 7 dex but only for 5500 K<Teff<14,000 K

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

    flam = np.zeros_like(x_shifted) * np.nan

    in_bounds = (x_shifted > 3600) & (x_shifted < 9000)
    flam[in_bounds] = np.interp(x_shifted[in_bounds], base_wavl_da, warwick_da_interp_low_logg((teff, logg)))

    #NICOLE BUG FIX
    norm=np.nanmedian(flam)
    flam = flam / norm # bring to order unity

    #NICOLE BUG FIX
    if norm==0:
        print("Median flux is 0. The fit has moved outside of the valid regime of the Montreal Models. These models cannot handle WDs with logg<7.5 and Teff<5000 K or >14000K")
        print("Tried Teff= "+str(teff)+" K, Logg="+str(logg)+" dex")
        
    dx = np.median(np.diff(x))
    window = res / dx
    
    flam = scipy.ndimage.gaussian_filter1d(flam, window)
    
    return flam

#NICOLE BUG FIX
def make_warwick_da_model_low_logg(resolution = 1, centres = default_centres, 
                       windows = default_windows, 
                       edges = default_edges,
                       names = default_names):
    """
    For loggs below 7 dex but only for 5500 K<Teff<14,000 K

    Parameters
    ----------
    resolution : float, optional
        gaussian sigma in AA by which the models are convolved. 
        The default is 1.
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
    
    model = Model(get_warwick_da_low_logg,
                  independent_vars = ['x'],
                  param_names = ['teff', 'logg', 'RV', 'res'])
    
    model.set_param_hint('teff', min = 5501, max = 14000, value = 12000)
    model.set_param_hint('logg', min = 5, max = 9, value = 8)
    model.set_param_hint('RV', min = -2500, max = 2500, value = 0)
    model.set_param_hint('res', value = resolution, min = 0, vary = False)
    
    
    model.centres = centres
    model.windows = windows
    model.names = names
    model.edges = edges
    
    return model



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
    
    nwl, nfl, _ = utils.cont_norm_lines(wl, flux, flux,
                                  corvmodel.names,
                                  corvmodel.centres,
                                  corvmodel.windows,
                                  corvmodel.edges)
    
    return nwl, nfl
