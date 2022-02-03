import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from astropy.table import Table
from tqdm import tqdm
import socket

hostname = socket.gethostname()
import multiprocessing
from multiprocessing import Pool
import sys

#plt.style.use('vedant')

import corv

if hostname[:4] == 'holy':
    #print('using holyoke paths')
    datapath = '/n/holyscratch01/conroy_lab/vchandra/wd/6_0_4/' # abs. path with CATID folders
    catpath = '/n/home03/vchandra/wd/01_ddwds/cat/' # abs. path with CATID folders
else:
    #print('using local paths')
    datapath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/data/ddcands/' # abs. path with CATID folders
    catpath = '/Users/vedantchandra/0_research/01_sdss5/006_build_corv/cat/' # abs. path with CATID folders

try:
    starcat = Table.read(catpath + 'starcat.fits')
    #expcat = Table.read(catpath + 'expcat.fits')

    #print('starcat has %i stars' % len(starcat))
except:
    pass#print('star and exposure catalogs not found! check paths and run make_catalogs() if you want to use sdss functionality. otherwise ignore.')

dacat = Table.read(catpath + 'dacat.fits')

keepcol = ['AIRMASS', 'ALT', 'AZ', 'DATE-OBS', 'DEC', 'EXPTIME', 'G_DR2', 'HELIO_RV', 
           'IPA', 'MJD', 'PLATEID', 'QUALITY', 'RA', 'SDSSNAME', 'SRVYMODE', 'TAI-BEG',
           'TAI-END', 'cid', 'expid', 'rfile', 'bfile']

kmodel7 = corv.models.make_koester_model(names = ['n', 'z', 'e', 'd', 'g', 'b', 'a'])
kmodel4 = corv.models.make_koester_model(names = ['d', 'g', 'b', 'a'])
bmodel = corv.models.make_balmer_model(names = ['n', 'z', 'e', 'd', 'g', 'b', 'a'])

def full_fit_corv(cid):

    star = dacat[dacat['cid'] == cid]

    star_header = dict(star[0])

    exps = corv.sdss.get_exposures(cid)

    wl, fl, ivar = corv.sdss.make_coadd(exps, method = 'ivar_mean')

    nexp = len(exps['header'])

    try:

        coadd_param_res, coadd_rv_res, coadd_rv_init = corv.fit.fit_corv(wl, fl, ivar, 
                                                                     kmodel7, iter_teff = True)

        coadd_param_res_b, coadd_rv_res_b, coadd_rv_init_b = corv.fit.fit_corv(wl, fl, ivar, 
                                                                     bmodel, iter_teff = False)
    except:
        print('coadd fit failed!')
        return [];

    star_header['coadd_teff'] = coadd_param_res.params['teff'].value
    star_header['coadd_teff_err'] = coadd_param_res.params['teff'].stderr

    star_header['coadd_logg'] = coadd_param_res.params['logg'].value
    star_header['coadd_logg_err'] = coadd_param_res.params['logg'].stderr

    star_header['coadd_rv_init_k'] = coadd_rv_init
    star_header['coadd_rv_k'] = coadd_rv_res.params['RV'].value
    star_header['coadd_rv_err_k'] = coadd_rv_res.params['RV'].stderr

    star_header['coadd_rv_init_b'] = coadd_rv_init_b
    star_header['coadd_rv_b'] = coadd_rv_res_b.params['RV'].value
    star_header['coadd_rv_err_b'] = coadd_rv_res_b.params['RV'].stderr

    ret_star = [];

    for expnum in range(nexp):
        exp_header = dict(exps['header'][expnum][keepcol])
        data = exps['data'][expnum]

        wl_i, fl_i, ivar_i = 10**data['logwl'], data['fl'], data['ivar']

        wlsel = (wl_i > 3600) & (wl_i < 9000)
        wl_i, fl_i, ivar_i = wl_i[wlsel], fl_i[wlsel], ivar_i[wlsel]

        try:
            exp_res_k, exp_rv_init_k = corv.fit.fit_rv(wl_i, fl_i, ivar_i, kmodel4, coadd_param_res.params)
            exp_res_b, exp_rv_init_b = corv.fit.fit_rv(wl_i, fl_i, ivar_i, bmodel, coadd_param_res_b.params)
        except:
            print('exposure fit failed!')
            continue

        exp_header['rv_init_k'] = exp_rv_init_k
        exp_header['rv_k'] = exp_res_k.params['RV'].value
        exp_header['rv_err_k'] = exp_res_k.params['RV'].stderr

        exp_header['rv_init_b'] = exp_rv_init_b
        exp_header['rv_b'] = exp_res_b.params['RV'].value
        exp_header['rv_err_b'] = exp_res_b.params['RV'].stderr

        full_header = {**star_header, **exp_header}

        ret_star.append(full_header)
        
    return ret_star

if __name__ == '__main__':

    #REPLACE THIS WITH POOL MAP

    # corv_data = [];

    # for star in tqdm(dacat):
    #     ret_star = full_fit_corv(star['cid'])
    #     corv_data.append(ret_star)
    #     break
    # print('finished!')

    n_cpu = int(sys.argv[1])
    print('there are %i CPU cores' % n_cpu)

    if bool(int(sys.argv[2])): # TEST OR NOT TEST
        dacat = dacat[:10]
        print('entering test mode, only fitting 10 stars')

    with Pool(n_cpu) as pool:

        corv_fits = list(tqdm(pool.imap(full_fit_corv, dacat['cid'])))

    corv_fits_flat = [item for sublist in corv_fits for item in sublist]

    print(corv_fits_flat)

    corvcat = Table(corv_fits_flat)

    corvcat = corvcat.filled(99.0)

    print(corvcat)

    corvcat.write(catpath + 'corvcat.fits', overwrite = True)