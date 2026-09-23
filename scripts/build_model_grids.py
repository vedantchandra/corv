#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the packaged model grids used by corv.models.GridModel.from_hdf5.

Parses the raw Koester-format model files (one file per logg, many Teff
spectra per file), puts every model on a common wavelength grid, crops to the
fitting range, converts to f_lambda and vacuum wavelengths, and arranges the
spectra into a regular (teff, logg, wavl) cube. Each model grid is written as
a group of a single HDF5 file:

    /<model_name>/wavl      (nwavl,)             vacuum wavelength [AA]
    /<model_name>/teff      (nteff,)             effective temperature [K]
    /<model_name>/logg      (nlogg,)             log10 surface gravity [cgs]
    /<model_name>/flux      (nteff, nlogg, nwavl) flux, missing models = -999
    /<model_name>/valid     (nteff, nlogg)       True where a model exists

Usage:
    python scripts/build_model_grids.py                   # build all grids
    python scripts/build_model_grids.py 1d_da_nlte        # rebuild one grid
    python scripts/build_model_grids.py --raw-dir DIR --output FILE
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_raw_dir = os.path.join(repo, 'src', 'corv', 'models')
default_output = os.path.join(repo, 'src', 'corv', 'models', 'corv_models.h5')

# model_name : (subdirectory of raw files, wavelength frame of raw files)
supported_models = {'1d_da_nlte': ('1d_da_nlte', 'air'),
                    '1d_elm_da_lte': ('1d_elm_da_lte', 'air'),
                    '3d_da_lte_noh2': ('3d_da_lte_noh2', 'vac'),
                    '3d_da_lte_h2': ('3d_da_lte_h2', 'vac'),
                    '3d_da_lte_old': ('3d_da_lte_old', 'air')}

wavl_range = (3600, 9000)
missing_value = -999

number_regex = "[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

def read_model_file(file):
    """
    Parse a single Koester-format model file.

    Returns
    -------
    wavl : ndarray (npoints,)
    params : list of (teff, logg)
    fluxes : list of ndarray (npoints,)
    """
    with open(file, 'r') as f:
        lines = f.read().split('\n')

    npoints = int(lines[0])
    headers = [i for i, line in enumerate(lines) if 'Effective' in line]

    wavl = np.array(' '.join(lines[1:headers[0]]).split(), dtype=float)
    assert len(wavl) == npoints, f"{file}: wrong number of wavelength points!"

    params, fluxes = [], []
    for n, indx in enumerate(headers):
        last = headers[n + 1] if n + 1 < len(headers) else len(lines)
        teff, gravity = [float(num) for num in re.findall(number_regex, lines[indx])[:2]]
        flux = np.array(' '.join(lines[indx + 1:last]).split(), dtype=float)
        assert len(flux) == npoints, f"{file}: wrong number of flux points at teff={teff}, g={gravity}!"
        params.append((teff, np.log10(gravity)))
        fluxes.append(flux)
    return wavl, params, fluxes

def air2vac(wavl):
    _tl = 1.e4 / wavl
    return wavl * (1. + 6.4328e-5 + 2.94981e-2 / (146. - _tl**2) + 2.5540e-4 / (41. - _tl**2))

def fnu_to_flam(wavl, fnu):
    return 2.99792458e18 * fnu / wavl**2

def build_grid(model_name, raw_dir):
    subdir, frame = supported_models[model_name]
    files = sorted(glob.glob(os.path.join(raw_dir, subdir, '*')))
    assert len(files) > 0, f'no raw model files found in {os.path.join(raw_dir, subdir)}'

    wavls, params, fluxes = [], [], []
    for file in files:
        wl, pars, fls = read_model_file(file)
        wavls += [wl] * len(fls)
        params += pars
        fluxes += fls

    # some grids mix wavelength sampling between files: put everything on the
    # finest (longest) grid
    reference = max(wavls, key=len)
    fluxes = np.array([fl if (len(wl) == len(reference) and np.array_equal(wl, reference))
                       else np.interp(reference, wl, fl)
                       for wl, fl in zip(wavls, fluxes)])

    mask = (wavl_range[0] < reference) & (reference < wavl_range[1])
    wavl, fluxes = reference[mask], fluxes[:, mask]

    # some raw grids repeat wavelength points: keep the first of each
    duplicate = np.r_[False, np.diff(wavl) <= 0]
    if duplicate.any():
        assert np.all(np.diff(wavl) >= 0), 'raw wavelength grid is not sorted'
        if not np.allclose(fluxes[:, duplicate], fluxes[:, np.roll(duplicate, -1)]):
            print('  warning: repeated wavelength points have different fluxes; keeping the first')
        print(f'  dropping {duplicate.sum()} repeated wavelength point(s)')
        wavl, fluxes = wavl[~duplicate], fluxes[:, ~duplicate]

    fluxes = fnu_to_flam(wavl, fluxes)
    if frame == 'air':
        wavl = air2vac(wavl)

    params = np.array(params)
    teff = np.unique(params[:, 0])
    logg = np.unique(params[:, 1])

    flux = np.full((len(teff), len(logg), len(wavl)), missing_value, dtype=float)
    valid = np.zeros((len(teff), len(logg)), dtype=bool)
    for (t, g), fl in zip(params, fluxes):
        i, j = np.searchsorted(teff, t), np.searchsorted(logg, g)
        if valid[i, j]:
            print(f'  warning: duplicate model at teff={t}, logg={g:.3f}; keeping the first')
            continue
        flux[i, j] = fl
        valid[i, j] = True

    return dict(wavl=wavl, teff=teff, logg=logg, flux=flux, valid=valid), frame, files

def write_grid(output, model_name, grid, frame, files, raw_dir):
    with h5py.File(output, 'a') as f:
        if model_name in f:
            del f[model_name]
        group = f.create_group(model_name)
        for key, value in grid.items():
            kwargs = dict(compression='gzip', shuffle=True) if value.ndim > 1 else {}
            group.create_dataset(key, data=value, **kwargs)
        group.attrs['units'] = 'flam'
        group.attrs['wavl_frame'] = 'vac'
        group.attrs['raw_wavl_frame'] = frame
        group.attrs['missing_value'] = missing_value
        group.attrs['source_files'] = [os.path.relpath(file, raw_dir) for file in files]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='*', default=list(supported_models),
                        help='model grids to build (default: all)')
    parser.add_argument('--raw-dir', default=default_raw_dir,
                        help='directory containing one subdirectory of raw files per model')
    parser.add_argument('--output', default=default_output, help='HDF5 file to write')
    args = parser.parse_args()

    for model_name in args.models:
        assert model_name in supported_models, f'{model_name} not in {list(supported_models)}'
        print(f'building {model_name}...')
        grid, frame, files = build_grid(model_name, args.raw_dir)
        write_grid(args.output, model_name, grid, frame, files, args.raw_dir)
        print(f"  {len(grid['teff'])} teff x {len(grid['logg'])} logg x {len(grid['wavl'])} wavl, "
              f"{(~grid['valid']).sum()} missing")
    print(f'wrote {args.output}')
