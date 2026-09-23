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
    - Add Koester DB models
"""

import numpy as np
from lmfit.models import Model, ConstantModel, VoigtModel, SkewedVoigtModel
import os
import scipy.ndimage
import h5py

from scipy.interpolate import RegularGridInterpolator
from . import utils

basepath = os.path.dirname(os.path.abspath(__file__))
default_grid_path = os.path.join(basepath, 'models', 'corv_models.h5')

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
                 names = default_names,
                 skewness = False):
    """
    Models each Balmer line as a (sum of) Voigt profiles

    Parameters
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
    model : LMFIT model
        LMFIT-style model that can be evaluated and fitted.

    """

    model = ConstantModel()
    for line in names:
        for n in range(nvoigt):
            if skewness:
                model -= SkewedVoigtModel(prefix = line + str(n) + '_')
            else:
                model -= VoigtModel(prefix = line + str(n) + '_')
    model.set_param_hint('c', value = 1)
    model.set_param_hint('RV', value = 0, min = -2500, max = 2500)
  
    for name in names:
        for n in range(nvoigt):
            pref = name + str(n)
            model.set_param_hint(pref + '_sigma', value = 15, min = 0)
            model.set_param_hint(pref + '_amplitude', value = 15, min = 0)
            if skewness:
                model.set_param_hint(pref + '_skew', value = 0, min = -1e-4)
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


def shift_and_broaden(x, RV, res, model_wavl, model_flux):
    """
    Doppler-shift a rest-frame template to velocity RV, sample it at x,
    bring it to order unity, and convolve it with a Gaussian.

    Parameters
    ----------
    x : array_like
        observed wavelengths in Angstrom.
    RV : float
        radial velocity in km/s.
    res : float
        Gaussian sigma of the instrumental broadening, in Angstrom.
    model_wavl, model_flux : array_like
        rest-frame template.

    Returns
    -------
    flam : array_like
        model flux at x, NaN outside 3600-9000 AA in the rest frame.
    """
    x_shifted = x * np.sqrt((1 - RV/c_kms)/(1 + RV/c_kms))

    flam = np.zeros_like(x_shifted) * np.nan
    in_bounds = (x_shifted > 3600) & (x_shifted < 9000)
    flam[in_bounds] = np.interp(x_shifted[in_bounds], model_wavl, model_flux)
    flam = flam / np.nanmedian(flam) # bring to order unity

    dx = np.median(np.diff(x))
    return scipy.ndimage.gaussian_filter1d(flam, res / dx)


class GridModel:
    """
    A (teff, logg) grid of model spectra wrapped as an LMFIT model with
    parameters teff, logg, RV and res. The LMFIT model is `self.model`.

    Parameters
    ----------
    interpolator : callable
        maps (teff, logg) to flux sampled on `wavl`.
    wavl : array_like
        rest-frame wavelengths of the interpolated spectra in Angstrom.
    teff_bounds, logg_bounds : tuple, optional
        fit bounds. Default to the extent of `interpolator.grid` when the
        interpolator has one (e.g. RegularGridInterpolator).
    resolution : float, optional
        Gaussian sigma in AA by which the models are convolved. The default is 1.
    centres, windows, edges, names : optional
        line definitions used for continuum normalization.

    The fit starts from teff = 12000, logg = 8 (clipped into the bounds).
    Use GridModel.from_hdf5 for the packaged grids and GridModel.from_koester
    for the Koester (2010) DA models.
    """
    def __init__(self, interpolator, wavl, teff_bounds = None, logg_bounds = None,
                 resolution = 1, centres = default_centres, windows = default_windows,
                 edges = default_edges, names = default_names):
        self.interpolator = interpolator
        self.wavl = wavl

        if teff_bounds is None or logg_bounds is None:
            assert hasattr(interpolator, 'grid'), \
                'teff_bounds and logg_bounds are required for interpolators without a .grid'
            teff_axis, logg_axis = interpolator.grid
            if teff_bounds is None:
                teff_bounds = (np.min(teff_axis), np.max(teff_axis))
            if logg_bounds is None:
                logg_bounds = (np.min(logg_axis), np.max(logg_axis))

        self.model = Model(self.evaluate, independent_vars = ['x'],
                           param_names = ['teff', 'logg', 'RV', 'res'])
        self.model.set_param_hint('teff', min = teff_bounds[0], max = teff_bounds[1],
                                  value = np.clip(12000, *teff_bounds))
        self.model.set_param_hint('logg', min = logg_bounds[0], max = logg_bounds[1],
                                  value = np.clip(8, *logg_bounds))
        self.model.set_param_hint('RV', min = -2500, max = 2500, value = 0)
        self.model.set_param_hint('res', value = resolution, min = 0, vary = False)

        self.model.centres = centres
        self.model.windows = windows
        self.model.names = names
        self.model.edges = edges

    def evaluate(self, x, teff, logg, RV, res):
        """
        Interpolates the grid, then shifts and broadens it (see shift_and_broaden).

        Parameters
        ----------
        x : array_like
            wavelength in Angstrom.
        teff : float
            effective temperature in K.
        logg : float
            log surface gravity in cgs.
        RV : float
            radial velocity in km/s.
        res : float
            Gaussian sigma of the instrumental broadening, in Angstrom.

        Returns
        -------
        flam : array_like
            synthetic flux interpolated at the requested parameters.
        """
        return shift_and_broaden(x, RV, res, self.wavl, self.interpolator((teff, logg)))

    @classmethod
    def from_hdf5(cls, model_name = '1d_da_nlte', grid_path = default_grid_path,
                  teff_bounds = None, logg_bounds = None, **kwargs):
        """
        Load a grid from the HDF5 file built by scripts/build_model_grids.py.
        Bounds default to the grid's extent, with teff floored at 4001 K. The
        loaded ModelGrid is kept as `self.grid`.
        """
        grid = ModelGrid(model_name, path = grid_path)
        if teff_bounds is None:
            teff_bounds = (max(grid.teff.min(), 4001), grid.teff.max())
        if logg_bounds is None:
            logg_bounds = (grid.logg.min(), grid.logg.max())
        obj = cls(grid.model_spec, grid.wavl, teff_bounds, logg_bounds, **kwargs)
        obj.grid = grid
        return obj

    @classmethod
    def from_koester(cls, teff_bounds = (3001, 39999), logg_bounds = (4.51, 9.49), **kwargs):
        """Koester (2010) DA models, via the optional `koester` package."""
        try:
            import koester
        except ImportError:
            raise ImportError('Koester models require the `koester` package. '
                              'Contact arseneau@bu.edu if these are needed.') from None
        interp = koester.WDInterpolator()
        return cls(interp.model_spec, interp.wavl_grid, teff_bounds, logg_bounds, **kwargs)


# backwards-compatible names

def WarwickDAModel(model_name = '1d_da_nlte', **kwargs):
    """Equivalent to GridModel.from_hdf5."""
    return GridModel.from_hdf5(model_name, **kwargs)

def make_koester_model(resolution = 1, **kwargs):
    """Equivalent to GridModel.from_koester(...).model"""
    return GridModel.from_koester(resolution = resolution, **kwargs).model


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

class ModelGrid:
    """
    Regular (teff, logg) grid of model spectra, loaded from the packaged HDF5
    file built by scripts/build_model_grids.py.
    """
    def __init__(self, model, path = default_grid_path):
        with h5py.File(path, 'r') as f:
            assert model in f, f'requested model not supported, choose from {list(f.keys())}'
            group = f[model]
            self.wavl = group['wavl'][:]
            self.teff = group['teff'][:]
            self.logg = group['logg'][:]
            self.flux_grid = group['flux'][:]
            self.valid = group['valid'][:]
            self.units = group.attrs['units']
        self.modelname = model

        self.model_spec = RegularGridInterpolator((self.teff, self.logg), self.flux_grid)
