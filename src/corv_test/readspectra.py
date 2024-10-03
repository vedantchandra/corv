import os
import re
import glob

import numpy as np
from scipy.interpolate import RegularGridInterpolator

class Spectrum:
    def __init__(self, model, units = 'flam'):
        supported_models = {'3d_da_lte_old': ('models/3d_da_lte/*', 2),
                            '1d_da_nlte': ('models/1d_da_nlte/*', 2)}
        assert model in list(supported_models.keys()), 'requested model not supported'
        # load in the model files
        dirname = os.path.dirname(os.path.abspath(__file__)) 
        self.path = os.path.join(dirname, supported_models[model][0])
        self.files = glob.glob(self.path)
        self.units = units
       
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
            self.fluxes = np.array(fluxes, dtype=float)
            wavls = np.array(wavls, dtype=float)
        except ValueError:
            wavls, self.fluxes = self.interpolate(wavls, fluxes, max(wl_grid_length))
        self.wavl = wavls[0]

        # convert to flam if that option is specified
        if self.units == 'flam':
            for i in range(len(self.fluxes)):
                self.fluxes[i] = 2.99792458e18 * self.fluxes[i] / wavls[i]**2 
      
        self.build_interpolator()


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
