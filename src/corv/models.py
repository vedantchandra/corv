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
from lmfit.models import Model, ConstantModel, VoigtModel, SkewedVoigtModel
import pickle
import os
import scipy 
import re
import glob

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from . import utils

try:
    import koester
    da_interp = koester.WDInterpolator()
except:
    print('Could not import Koester models. Contact arseneau@bu.edu if these are needed.')

basepath = os.path.dirname(os.path.abspath(__file__))
modpath = basepath[:-8] + 'models/'

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

# Koester DA Model
#if modpath!='no path selected':
#    try:
#        print(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'koester_interp_da.pkl'))
#        wd_interp = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'koester_interp_da.pkl'), 'rb'))
#    except:
#        print('Could not find the pickled WD models. If you need to use these models, please re-import corv with the proper path.')

def get_koester(x, teff, logg, RV, res):
    """Interpolates Koester (2010) DA models

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
    #flam[in_bounds] = 10**wd_interp((logg, np.log10(teff), np.log10(x_shifted[in_bounds])))
    #koester_interp = koester.WDInterpolator()
    flam[in_bounds] = np.interp(x_shifted[in_bounds], da_interp.wavl_grid, da_interp.model_spec((teff, logg)))

    flam = flam / np.nanmedian(flam) # bring to order unity
    
    dx = np.median(np.diff(x))
    window = res / dx
    
    flam = scipy.ndimage.gaussian_filter1d(flam, window)
    
    return flam


def make_koester_model(resolution = 1, centres = default_centres, 
                       windows = default_windows, 
                       edges = default_edges,
                       names = default_names):
    """Parameters
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
## UPDATED INTERPOLATION & MODEL SUPPORT
class WarwickDAModel:
    def __init__(self, model_name = '1d_da_nlte', resolution = 1, 
                       centres = default_centres, windows = default_windows, 
                       edges = default_edges, names = default_names):

        self.interpolator = Spectrum(model_name)
        self.model = Model(self.get_warwick, independent_vars = ['x'],
                  param_names = ['teff', 'logg', 'RV', 'res'])

        if model_name == '3d_da_lte_h2':
            self.model.set_param_hint('teff', min = 4001, max = 39900, value = 12000)
        else:
            self.model.set_param_hint('teff', min = 4001, max = 129000, value = 12000)
        self.model.set_param_hint('logg', min = 7, max = 9, value = 8)
        self.model.set_param_hint('RV', min = -2500, max = 2500, value = 0)
        self.model.set_param_hint('res', value = resolution, min = 0, vary = False)
    
        self.model.centres = centres
        self.model.windows = windows
        self.model.names = names
        self.model.edges = edges

    def get_warwick(self, x, teff, logg, RV, res):
        """
        Interpolates Montreal models

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
        flam[in_bounds] = np.interp(x_shifted[in_bounds], self.interpolator.wavl, self.interpolator.model_spec((teff, logg)))
        #flam[in_bounds] = self.interpolator.model_spec((teff, logg, x_shifted[in_bounds]))
        #raise "aaaaaaa"
        norm = np.nanmedian(flam)
        flam = flam / norm # bring to order unity
            
        dx = np.median(np.diff(x))
        window = res / dx
        
        flam = scipy.ndimage.gaussian_filter1d(flam, window)
        return flam


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

class Spectrum:
    def __init__(self, model, units = 'flam', wavl_range = (3600, 9000)):
        # (path_to_files, n_free_parameters, wavlength_frame)
        supported_models = {'1d_da_nlte': ('models/1d_da_nlte/*', 2, 'air'),
                            '1d_elm_da_lte': ('models/1d_elm_da_lte/*', 2, 'air'),
                            '3d_da_lte_noh2': ('models/3d_da_lte_noh2/*', 2, 'vac'),
                            '3d_da_lte_h2': ('models/3d_da_lte_h2/*', 2, 'vac'),
                            '3d_da_lte_old': ('models/3d_da_lte_old/*', 2, 'air')}
        assert model in list(supported_models.keys()), 'requested model not supported'
        # load in the model files
        dirname = os.path.dirname(os.path.abspath(__file__)) 
        self.path = os.path.join(dirname, supported_models[model][0])
        self.files = glob.glob(self.path)
        self.units = units
        self.modelname = model
        self.wavl_range = wavl_range
       
        # fetch the wavelength and flux grids
        self.nparams = supported_models[model][1]
        wavls, values, fluxes = [], [], []
        for file in self.files:
            wl, vals, fls = self.filehandler(file)
            wavls += wl
            values += vals
            fluxes += fls
        
        self.values = np.array(values, dtype=float)
        wl_grid_length = list(set([len(wl) for wl in wavls]))
        # Handle multiple wavelength grids in the same model
        try:
            fluxes_np = np.array(fluxes, dtype=float)
            wavls = np.array(wavls, dtype=float)
        except ValueError:
            wavls, fluxes_np = self.interpolate(wavls, fluxes, max(wl_grid_length))
        
        mask = (self.wavl_range[0] < wavls[0]) & (wavls[0] < self.wavl_range[1])
        self.wavl, self.fluxes = wavls[0][mask], fluxes_np[:,mask]

        # convert to flam if that option is specified
        if self.units == 'flam':
            for i in range(len(self.fluxes)):
                self.fluxes[i] = 2.99792458e18 * self.fluxes[i] / self.wavl[i]**2 

        if supported_models[model][2] == 'air':
            self.air2vac()
      
        self.build_interpolator()

    def air2vac(self):
        _tl=1.e4/self.wavl
        self.wavl = (self.wavl*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))


    def interpolate(self, wavls, fluxes, length_to_interpolate):
        for i in range(len(wavls)):
            if len(wavls[i]) == length_to_interpolate:
                reference_grid = wavls[i]
                break
        for i in range(len(wavls)):
            if len(wavls[i]) != length_to_interpolate:
                fluxes[i] = np.interp(reference_grid, wavls[i], fluxes[i])
                wavls[i] = reference_grid
        return np.array(wavls), np.array(fluxes)

    def build_interpolator(self):
        self.unique_logteff = np.array(sorted(list(set(self.values[:,0]))))
        self.unique_logg = np.array(sorted(list(set(self.values[:,1]))))
        self.flux_grid = np.zeros((len(self.unique_logteff), 
                                len(self.unique_logg), 
                                len(self.wavl)))

        for i in range(len(self.unique_logteff)):
            for j in range(len(self.unique_logg)):
                target = [self.unique_logteff[i], self.unique_logg[j]]
                try:
                    indx = np.where((self.values == target).all(axis=1))[0][0]
                    self.flux_grid[i,j] = self.fluxes[indx]
                except IndexError:
                    self.flux_grid[i,j] += -999

        self.model_spec = RegularGridInterpolator((10**self.unique_logteff, self.unique_logg), self.flux_grid) 

    def filehandler(self, file):
        with open(file, 'r') as f:
            fdata = f.read()

        wavl = self.fetch_wavl(fdata)
        values, fluxes = self.fetch_spectra(fdata)
        dim_wavl = []
        for i in range(len(fluxes)):
            dim_wavl.append(wavl)
        return dim_wavl, values, fluxes

    def fetch_wavl(self, fdata):
        def get_linenum(npoints):
            first = 1
            last = (npoints // 10 + 1)
            if (npoints % 10) != 0:
                last += 1
            return first, last
        # figure out how many points are in the file
        lines = fdata.split('\n')
        npoints = int(lines[0])
        first, last = get_linenum(npoints)

        wavl = []
        for line in lines[first:last]:
            line = line.strip('\n').split()
            for num in line:
                wavl.append(float(num))
        return wavl

    def fetch_spectra(self, fdata):
        def idx_to_params(indx, first_n):
            string = lines[indx]
            regex = "[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
            params = re.findall(regex, string)[:first_n]
            params = [np.log10(float(num)) for num in params]
            return params
        lines = fdata.split('\n')
        npoints = int(lines[0])
        indx = [i for i, line in enumerate(lines) if 'Effective' in line]
        
        values, fluxes = [], []
        for n in range(len(indx)):
            first = indx[n]+1
            try:
                last = indx[n+1]
            except IndexError:
                last = len(lines)
            params = idx_to_params(indx[n], self.nparams)
            values.append(params)

            flux = []
            for line in lines[first:last]:
                line = line.strip('\n').split()
                for num in line:
                    flux.append(float(num))

            assert len(flux) == npoints, "Error reading spectrum: wrong number of points!"
            fluxes.append(flux)
        return values, fluxes
