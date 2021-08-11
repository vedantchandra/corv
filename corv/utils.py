#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:08:00 2021

@author: vedantchandra
"""

import numpy as np
from bisect import bisect_left


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
    window : int
        selected region on either side of line, in pixels.
    edge : int
        number of pixels on edge of region used to define continuum.

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

    mask = np.ones(len(wl))
    mask[edge:-edge] = 0
    mask = mask.astype(bool)

    p = np.polynomial.polynomial.polyfit(wl[mask], fl[mask], 1)
    continuum = np.polynomial.polynomial.polyval(wl, p)
    norm_fl = fl / continuum
    norm_ivar = ivar * continuum**2
    return wl, norm_fl, norm_ivar

def air2vac(wv):
    """
    Air to vacuum wavelengths, formula from Morton 1991 ApJS, 77, 119.

    Parameters
    ----------
    wv : array_like
        air wavelengths in Angstrom.

    Returns
    -------
    arary_like
        vacuum wavelengths in Angstrom. 

    """
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

def vac2air(wv):
    """
    Vacuum to air wavelengths, formula from Morton 1991 ApJS, 77, 119.

    Parameters
    ----------
    wv : array_like
        vacuum wavelengths in Angstrom.

    Returns
    -------
    arary_like
        air wavelengths in Angstrom. 

    """
    _tl = 1.e4/np.array(wv)
    return (np.array(wv) / (1. + 6.4328e-5 + 2.94981e-2
                            / (146. - _tl**2) + 2.5540e-4 / (41. - _tl**2)))
        
