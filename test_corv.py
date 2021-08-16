#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:47:04 2021

@author: vedantchandra
"""

from matplotlib import pyplot as plt
import numpy as np

import corv

# def test_travis():
#     wl = np.linspace(4000, 8000, 8000)

#     plt.figure(figsize = (7, 7))
#     corvmodel = corv.models.make_balmer_model(names = ['a'])
#     params = corvmodel.make_params()
#     nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
#     plt.plot(nwl, nfl, 'k.')
#     params['RV'].set(value = 1000)
#     nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
#     plt.plot(nwl, nfl, 'r.')
#     plt.title('Voigt Balmer')

# test_travis()


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

wl = np.linspace(4000, 8000, 8000)
corvmodel = corv.models.make_koester_model()
params = corvmodel.make_params()
params['RV'].set(value = 132)
fl = corvmodel.eval(params, x = wl)
sig = 0.1 * np.median(fl)
fl += sig * np.random.normal(size = len(wl))
ivar = 0*fl + 1 / (sig)**2


param_res, rv_res, rv_init = corv.fit.fit_corv(wl, fl, ivar, corvmodel)

print(rv_init)
print(param_res.params['RV'])
print(rv_res.params['RV'])