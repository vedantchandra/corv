#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:52:06 2021

These scripts test the RV recovery ability, as a function of several 
parameters (S/N, etc). It should help guide the development of the 
RV fitting scheme. 

@author: vedantchandra
"""

import corv

# RV and RV error vs S/N

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