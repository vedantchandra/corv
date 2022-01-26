#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:47:04 2021

@author: vedantchandra
"""

from matplotlib import pyplot as plt
import numpy as np

import corv

def test_travis():
    wl = np.linspace(4000, 8000, 8000)

    plt.figure(figsize = (7, 7))
    corvmodel = corv.models.make_balmer_model(names = ['a'])
    params = corvmodel.make_params()
    nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
    plt.plot(nwl, nfl, 'k.')
    params['RV'].set(value = 1000)
    nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
    plt.plot(nwl, nfl, 'r.')
    plt.title('Voigt Balmer')
    plt.close()

test_travis()

# wl = np.linspace(4000, 8000, 8000)
# plt.figure(figsize = (7,7))
# corvmodel = corv.models.make_koester_model(names = ['a'])
# params = corvmodel.make_params()
# nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
# plt.plot(nwl, nfl, 'k.')
# params['RV'].set(value = 1000)
# params['teff'].set(value = 25500)
# nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
# plt.plot(nwl, nfl, 'r.')
# plt.title('Koester DA')