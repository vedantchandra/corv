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
import astropy

#plt.style.use('vedant')

import corv
from corv.sdss import get_ew_dict

if __name__ == '__main__':


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

		starcat = starcat[0:10]

		expcat = Table.read(catpath + 'expcat.fits')
		#print('starcat has %i stars' % len(starcat))
	except:
		pass#print('star and exposure catalogs not found! check paths and run make_catalogs() if you want to use sdss functionality. otherwise ignore.')


	n_cpu = multiprocessing.cpu_count()
	print('there are %i CPU cores' % n_cpu)

	ewdicts = [];

	with Pool(n_cpu) as pool:

		ewdicts = pool.map(get_ew_dict, starcat['cid'])

	ewtable = Table(ewdicts)

	ewcat = astropy.table.join(starcat, ewtable, keys = 'cid')

	ewcat.write(catpath + 'ewcat.fits', overwrite = True)