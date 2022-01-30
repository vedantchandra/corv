#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:06:32 2021

Functions specific to SDSS-5, including data processing. 

@author: vedantchandra
"""

# SPIN OFF THESE PATHS TO SWITCH BASED ON WHETHER LAPTOP OR HOLY

import numpy as np
from astropy.io import fits
import glob
from astropy.table import Table
from tqdm import tqdm
import socket
hostname = socket.gethostname()

from . import spectral_resampling

if hostname[:4] == 'holy':
	print('using holyoke paths')
	datapath = '/n/holyscratch01/conroy_lab/vchandra/wd/6_0_4/' # abs. path with CATID folders
	catpath = '/n/home03/vchandra/wd/01_ddwds/cat/' # abs. path with CATID folders
else:
	print('using local paths')
	datapath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/data/ddcands/' # abs. path with CATID folders
	catpath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/cat/' # abs. path with CATID folders

try:
	starcat = Table.read(catpath + 'starcat.fits')
	expcat = Table.read(catpath + 'expcat.fits')
except:
	print('star and exposure catalogs not found! check paths and run make_catalogs() if you want to use sdss functionality. otherwise ignore.')

####### I/O FUNCTIONS #############

# Construct exp and exps data structures: dictionaries that contain all the required data

# From SDSS-V co-add datamodel
# There are the final wavelengths the co-adds are interpolated on

loglam0 = 3.55230
step = 0.0001
nwl = 4648
loglamgrid = np.linspace(loglam0, loglam0 + nwl * step, nwl)
lamgrid = 10**loglamgrid
break_wl = 5900
log_break_wl = np.log10(break_wl)

def get_exposure(bf, rf):
	
	# Given a red and blue single exposure .fits file, return `exp` object
	
	exp = {};

	with fits.open(bf) as f:
		
		exp['header'] = dict(f[0].header)
		
		bwl = f[3].data
		bfl = f[0].data
		bivar = f[1].data
		bdisp = f[4].data
		bsky = f[5].data
		bmask = f[2].data
		
	with fits.open(rf) as f:
		
		rwl = f[3].data
		rfl = f[0].data
		rmask = f[2].data
		rivar = f[1].data
		rdisp = f[4].data
		rsky = f[5].data
		
	rsel = rwl > log_break_wl
	bsel = bwl < log_break_wl
		
	exp['logwl'] = np.concatenate((bwl[bsel], rwl[rsel])).astype(float)
	exp['fl'] = np.concatenate((bfl[bsel], rfl[rsel])).astype(float)
	exp['ivar'] = np.concatenate((bivar[bsel], rivar[rsel])).astype(float)
	exp['wdisp'] = np.concatenate((bdisp[bsel], rdisp[rsel])).astype(float)
	exp['sky'] = np.concatenate((bsky[bsel], rsky[rsel])).astype(float)
	exp['mask'] = np.concatenate((bmask[bsel], rmask[rsel])).astype(float)
		
	return exp

def get_exposures(catalogid):
	
	# Given a catalogid, get all exposures. Return exp data and table
	
	seltable = expcat[expcat['cid'] == catalogid]
	expsdata = [];
	expsheader = [];
	
	seltable = seltable[np.argsort(seltable['TAI-BEG'])]
	
	for ii,row in enumerate(seltable):
		exp = get_exposure(row['bfile'], row['rfile'])
		exp.pop('header') # no need to duplicate what's in the table already
		expsdata.append(exp)
			
	exps = dict(data = expsdata, header = seltable)
	
	return exps

def make_coadd(exps, method = 'ivar_mean'):
	
	# Todo: add WDISP here too
	# Todo: some way to pass velocities here to shift and then co-add
	# Todo: add different coadding methods: median, inverse variance mean. 
	
	nexp = len(exps['data'])

	fls = np.zeros((nexp, nwl))
	sigmas = np.zeros((nexp, nwl))

	for ii,exp in enumerate(exps['data']):

		with np.errstate(divide='ignore'):
			fl_i, sigma_i = spectral_resampling.spectres(loglamgrid, exp['logwl'], exp['fl'], np.reciprocal(np.sqrt(exp['ivar'])))

		fls[ii, :] = fl_i
		sigmas[ii, :] = sigma_i

	fl = np.zeros(nwl)
	sigma = np.zeros(nwl)
	mask = np.ones(nwl).astype(bool)
	#nexps = np.zeros(nwl)

	for px in range(nwl):
				
		fl_px = fls[:, px]
		sigma_px = sigmas[:, px]
		
		# sigma clipping? disabled for now.. 
#         fl_px_clipped, lb, ub = stats.sigmaclip(fl_px, low = 5.0, high = 5.0)
#         sigma_px = sigma_px[(fl_px > lb) & (fl_px < ub)]
#         fl_px = fl_px_clipped
						
		goodmask = np.isfinite(sigma_px)
		ngood = np.sum(goodmask)
		#nexps[px] = ngood

		if ngood > 0:

			if method == 'median':
				fl[px] = np.median(fl_px[goodmask])
				sigma[px] = np.sqrt(np.sum(sigma_px[goodmask]**2)) / ngood # CHECK MATH HERE, ERROR ON MEDIAN? 
			elif method == 'mean':
				fl[px] = np.mean(fl_px[goodmask])
				sigma[px] = np.sqrt(np.sum(sigma_px[goodmask]**2)) / ngood
			elif method == 'ivar_mean':
				fl[px] = np.average(fl_px[goodmask], weights = 1/sigma_px[goodmask]**2)
				sigma[px] = np.sqrt(1 / np.sum(1/sigma_px[goodmask]**2))
			else:
				print('invalid method!')

		else:
			mask[px] = False # where mask == False, there is zero usable data
			fl[px] = np.nan
			sigma[px] = np.inf
			
	ivar = 1 / sigma**2
	ivar[~mask] = 0.0
	
	return lamgrid, fl, ivar

####### CATALOG CONSTRUCTION #############


def make_catalogs(make_filecat = True, make_expcat = True, make_starcat = True):

	# Make filecat, expcat, and starcat given the datapath and output catpath. 

	#### Make `filecat.fits`

	if make_filecat:

		print('making %sfilecat.fits' % catpath)

		fits_files = glob.glob(datapath + '/*/*.fits')
		filecat_rows = []; 

		for nn,file in enumerate(tqdm(fits_files)):
			
			with fits.open(file) as f:
			
				row = dict(f[0].header)
				row['filepath'] = file
				row['cid'] = int(file.split('-')[-3])
				row['camera'] = file.split('-')[-2]
				row['expid'] = int(file.split('-')[-1].split('.')[0])
				
			filecat_rows.append(row)
			

		filecat = Table(filecat_rows)
		print('filecat has %i rows' % len(filecat))
		filecat.write(catpath + 'filecat.fits', overwrite = True)

	#### Make `expcat.fits`

	if make_expcat:

		mjds = np.unique(filecat['MJD'])

		expcat_rows = [];

		for mjd in tqdm(mjds):

		    mjdtable = filecat[filecat['MJD'] == mjd]
		    
		    cids = np.unique(mjdtable['cid'])
		    
		    for cid in cids:
		        
		        cidtable = mjdtable[mjdtable['cid'] == cid]

		        expids = np.unique(cidtable['expid'])

		        for expid in (expids):

		            seltab = cidtable[cidtable['expid'] == expid]

		            print(len(seltab))

		            row = dict(seltab[0])
		            row.pop('camera')
		            row.pop('filepath')

		            redfile = seltab[seltab['camera'] == 'r1']['filepath'][0]
		            bluefile = seltab[seltab['camera'] == 'b1']['filepath'][0]

		            row['rfile'] = redfile
		            row['bfile'] = bluefile

		            row['dr2_id'] = int(seltab['G_DR2'][0])

		            expcat_rows.append(row)

		expcat = Table(expcat_rows)
		print('expcat has %i rows' % len(expcat))
		expcat.write(catpath + 'expcat.fits', overwrite = True)

	#### Make `starcat.fits`

	if make_starcat:

		print('making %sstarcat.fits' % catpath)

		cids = np.unique(expcat['cid'])
		starcat_rows = [];

		for cid in tqdm(cids):
			seltab = expcat[expcat['cid'] == cid]
			
			row = {}
			row['cid'] = cid
			row['nexp'] = len(seltab)
			row['dr2_id'] = seltab['dr2_id'][0]
			row['spec_ra'] = np.round(np.median(seltab['RA']), 5)
			row['spec_dec'] = np.round(np.median(seltab['DEC']), 5)
			
			starcat_rows.append(row)

		starcat = Table(starcat_rows)
		print('starcat has %i rows' % len(starcat))
		starcat.write(catpath + 'starcat.fits', overwrite = True)
