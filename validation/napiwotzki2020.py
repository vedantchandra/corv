#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate corv RVs against the SPY survey (Napiwotzki et al. 2020, A&A 638,
A131; VizieR J/A+A/638/A131).

SPY measured multi-epoch UVES radial velocities for 643 DA white dwarfs. This
script:
    1. downloads the SPY tables from VizieR and keeps stars consistent with a
       constant RV (see select_constant);
    2. cross-matches them with SDSS DR17 spectra and downloads the spectra;
    3. fits each spectrum with corv.fit.fit_corv;
    4. compares the corv RV with the SPY weighted-mean RV.

Both SPY and corv fit the full line profile of a model at rest, so both RVs
include the gravitational redshift and can be compared directly. SPY and SDSS
velocities are both documented as heliocentric; frame_check tests this
empirically by regressing corv - SPY against each survey's heliocentric
correction.

Downloads are cached in validation/cache/. Run as

    python validation/napiwotzki2020.py

which writes validation/output/napiwotzki2020.ecsv and .png and prints a
summary. Requires astroquery (pip install "corv[validation]").
"""

import argparse
import os
import warnings

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.table import Table, join, unique
from astropy.time import Time
import astropy.units as u

import corv

here = os.path.dirname(os.path.abspath(__file__))
default_cache = os.path.join(here, 'cache')
default_output = os.path.join(here, 'output')

c_kms = 2.99792458e5
vizier_id = 'J/A+A/638/A131'
bad_remarks = ('magnetic', 'primary') # primary: parameters from a double-degenerate fit
paranal = EarthLocation.from_geodetic(lon = -70.4045 * u.deg, lat = -24.6268 * u.deg,
                                      height = 2635 * u.m) # SPY used UVES at the VLT

### SPY CATALOG ###

def load_spy(cache = default_cache):
    """
    One row per SPY star: name, coordinates, weighted-mean RV and its error,
    constant-RV log probability, SPY teff/logg, and remarks.
    """
    path = os.path.join(cache, 'spy_stars.ecsv')
    if os.path.exists(path) and 'helio_spy' in Table.read(path).colnames: # else outdated
        stars = Table.read(path)
        # ECSV reads empty strings back as masked
        for col in ('n_logp', 'remark'):
            if hasattr(stars[col], 'filled'):
                stars[col] = stars[col].filled('')
        return stars

    from astroquery.vizier import Vizier
    os.makedirs(cache, exist_ok = True)
    tables = Vizier(row_limit = -1, columns = ['**']).get_catalogs(vizier_id)

    objects = unique(tables[vizier_id + '/objects'], keys = 'Name')
    coords = SkyCoord(objects['RAJ2000'], objects['DEJ2000'], unit = (u.hourangle, u.deg))
    objects = Table(dict(name = objects['Name'], ra = coords.ra.deg, dec = coords.dec.deg))

    # the mean RV and variability test are given on the first epoch of each star
    rvs = tables[vizier_id + '/tableb1']
    for col in ('HRV', '<RV>'):
        assert 'heliocentric' in rvs[col].description.lower(), \
            'SPY %s is not documented as heliocentric: %s' % (col, rvs[col].description)
    epochs = rvs[np.isfinite(np.ma.filled(rvs['e_HRV'], np.nan))]
    Table(dict(name = epochs['Name'], hjd = epochs['HJD'] + 2400000, hrv = epochs['HRV'],
               e_hrv = epochs['e_HRV'])).write(os.path.join(cache, 'spy_epochs.ecsv'),
                                               overwrite = True)
    rvs = rvs[~np.ma.getmaskarray(rvs['<RV>'])]
    rvs = Table(dict(name = rvs['Name'],
                     rv_spy = np.ma.filled(rvs['<RV>'], np.nan),
                     e_rv_spy = np.ma.filled(rvs['e_<RV>'], np.nan),
                     logp = np.ma.filled(rvs['logp'], np.nan),
                     n_logp = np.ma.filled(rvs['n_logp'], '')))

    params = tables[vizier_id + '/tablec2']
    params = Table(dict(name = params['Name'],
                        teff_spy = params['Teff'], logg_spy = params['logg'],
                        remark = np.ma.filled(params['Rem'], '')))

    stars = join(join(rvs, params, keys = 'name'), objects, keys = 'name')
    stars['helio_spy'] = spy_heliocentric_correction(stars, Table.read(
        os.path.join(cache, 'spy_epochs.ecsv')))
    stars.write(path, overwrite = True)
    return stars

def spy_heliocentric_correction(stars, epochs):
    """
    For each star, the heliocentric correction at Paranal averaged over its
    SPY epochs with the same 1/e_HRV^2 weights as SPY's mean RV.
    """
    out = np.full(len(stars), np.nan)
    for ii, star in enumerate(stars):
        ep = epochs[epochs['name'] == star['name']]
        if len(ep) == 0:
            continue
        coord = SkyCoord(star['ra'], star['dec'], unit = 'deg')
        # HJD vs. JD differs by < 9 minutes, negligible for the correction
        corr = coord.radial_velocity_correction('heliocentric', location = paranal,
                                                obstime = Time(ep['hjd'], format = 'jd'))
        out[ii] = np.average(corr.to_value(u.km / u.s), weights = 1 / ep['e_hrv']**2)
    return out

def select_constant(stars, logp_min = -3):
    """
    Stars with no sign of RV variability: at least two epochs, log p >= logp_min
    (SPY flags binaries at log p < -4), not flagged as double-lined, and not
    magnetic or part of a double-degenerate fit.
    """
    keep = np.isfinite(stars['logp']) & (stars['logp'] >= logp_min)
    keep &= np.array([flag.strip() == '' for flag in stars['n_logp']])
    keep &= np.array([not any(bad in rem for bad in bad_remarks) for rem in stars['remark']])
    return stars[keep]

### SDSS SPECTRA ###

def match_sdss(stars, cache = default_cache, radius = 3 * u.arcsec):
    """All SDSS DR17 spectra within radius of each star; one row per spectrum."""
    path = os.path.join(cache, 'spy_sdss.ecsv')
    if os.path.exists(path):
        matches = Table.read(path)
    else:
        from astroquery.sdss import SDSS
        spy = load_spy(cache)
        coords = SkyCoord(spy['ra'], spy['dec'], unit = 'deg')
        matches = SDSS.query_crossid(coords, spectro = True, radius = radius,
                                     data_release = 17,
                                     specobj_fields = ['plate', 'mjd', 'fiberID', 'z',
                                                       'zErr', 'instrument', 'subClass'])
        # query_crossid names inputs obj_<index>
        matches['name'] = [spy['name'][int(n.split('_')[1])] for n in matches['name']]
        matches = matches['name', 'plate', 'mjd', 'fiberID', 'instrument', 'subClass', 'z', 'zErr']
        os.makedirs(cache, exist_ok = True)
        matches.write(path)
    return join(stars, matches, keys = 'name')

def get_spectrum(plate, mjd, fiber, cache = default_cache):
    """
    Download (or read from cache) an SDSS spectrum. Returns vacuum wavelength,
    flux and inverse variance, with ivar = 0 on pixels flagged in and_mask,
    and the heliocentric correction the pipeline applied (km/s).
    """
    path = os.path.join(cache, 'spectra', 'spec-%04i-%05i-%04i.fits' % (plate, mjd, fiber))
    if not os.path.exists(path):
        from astroquery.sdss import SDSS
        hdul = SDSS.get_spectra(plate = plate, mjd = mjd, fiberID = fiber, data_release = 17)[0]
        os.makedirs(os.path.dirname(path), exist_ok = True)
        # write then rename, so concurrent runs never see a partial file
        tmp = '%s.%i.tmp' % (path, os.getpid())
        hdul.writeto(tmp, overwrite = True)
        os.replace(tmp, path)
    with fits.open(path) as hdul:
        header = hdul[0].header
        assert header['VACUUM'], 'SDSS wavelengths are not in vacuum'
        helio_rv = header['HELIO_RV'] # applied by the pipeline
        data = hdul[1].data
        wl = 10**data['loglam'].astype(float)
        fl = data['flux'].astype(float)
        ivar = np.where(data['and_mask'] == 0, data['ivar'], 0).astype(float)
    return wl, fl, ivar, helio_rv

### FITTING ###

def fit_spectrum(wl, fl, ivar, corvmodel, init_teffs = (10000, 20000),
                 rv_kw = dict(min_rv = -500, max_rv = 500, npoints = 501),
                 plot_stem = None, title = ''):
    """
    fit_corv on one spectrum; returns rv, e_rv, redchi, teff, logg. If
    plot_stem is given, saves the line fit to <plot_stem>_lines.png and the
    chi-square curve to <plot_stem>_chi2.png.
    """
    if plot_stem is not None:
        rv_kw = dict(rv_kw, plot = True, path = plot_stem + '_chi2.png')
    rv, e_rv, redchi, param_res = corv.fit.fit_corv(wl, fl, ivar, corvmodel,
                                                    init_teffs = init_teffs, rv_kw = rv_kw)
    if plot_stem is not None:
        import matplotlib.pyplot as plt
        fig = corv.utils.lineplot(wl, fl, ivar, corvmodel, param_res.params)
        fig.axes[0].set_title('%s corv %.1f +/- %.1f km/s' % (title, rv, e_rv))
        fig.savefig(plot_stem + '_lines.png')
        plt.close(fig)
    return rv, e_rv, redchi, param_res.params['teff'].value, param_res.params['logg'].value

def run(cache = default_cache, model_name = '1d_da_nlte', names = ('d', 'g', 'b', 'a'),
        logp_min = -3, verbose = True, plot = True):
    """
    Fit every SDSS spectrum of every constant-RV SPY star. Returns one row per
    spectrum with the SPY and corv RVs. Fits that fail get NaN. If plot, the
    line fit and chi-square curve of each spectrum are saved in <cache>/fits/.
    """
    model = corv.models.GridModel.from_hdf5(model_name, names = list(names))
    spectra = match_sdss(select_constant(load_spy(cache), logp_min), cache)
    plot_dir = os.path.join(cache, 'fits')
    if plot:
        os.makedirs(plot_dir, exist_ok = True)

    cols = ['rv', 'e_rv', 'redchi', 'teff', 'logg']
    results = {col: np.full(len(spectra), np.nan) for col in cols + ['helio_sdss']}
    for ii, row in enumerate(spectra):
        try:
            wl, fl, ivar, results['helio_sdss'][ii] = get_spectrum(row['plate'], row['mjd'],
                                                                   row['fiberID'], cache)
            if not np.any(ivar > 0):
                raise ValueError('every pixel is flagged in and_mask')
            spec_id = '%04i-%05i-%04i' % (row['plate'], row['mjd'], row['fiberID'])
            plot_stem = os.path.join(plot_dir, '%s_%s' % (row['name'], spec_id)) if plot else None
            title = '%s (%s): SPY %.1f +/- %.1f km/s,' % (row['name'], spec_id,
                                                               row['rv_spy'], row['e_rv_spy'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = fit_spectrum(wl, fl, ivar, model.model, plot_stem = plot_stem,
                                   title = title)
            for col, val in zip(cols, out):
                results[col][ii] = val
        except Exception as e:
            print('%s (%i-%i-%i) failed: %s' % (row['name'], row['plate'], row['mjd'],
                                                row['fiberID'], e))
        if verbose:
            print('[%i/%i] %s: corv %.1f +/- %.1f, SPY %.1f +/- %.1f km/s' %
                  (ii + 1, len(spectra), row['name'], results['rv'][ii],
                   results['e_rv'][ii], row['rv_spy'], row['e_rv_spy']))

    for col in results:
        spectra[col] = results[col]
    return spectra

### COMPARISON ###

def compare(rv, e_rv, rv_ref, e_ref):
    """
    Statistics of rv - rv_ref over finite entries: the median offset, robust
    scatter (1.4826 * MAD), and for the error-normalized differences
    z = (rv - rv_ref) / sqrt(e_rv^2 + e_ref^2), their robust scatter and the
    fraction with |z| < 3.
    """
    ok = np.isfinite(rv) & np.isfinite(e_rv) & np.isfinite(rv_ref) & np.isfinite(e_ref)
    delta = rv[ok] - rv_ref[ok]
    z = delta / np.hypot(e_rv[ok], e_ref[ok])
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    return dict(n = int(ok.sum()),
                median_offset = np.median(delta),
                e_median_offset = 1.253 * mad(delta) / np.sqrt(ok.sum()),
                scatter = mad(delta),
                median_error = np.median(np.hypot(e_rv[ok], e_ref[ok])),
                z_scatter = mad(z),
                frac_within_3sigma = np.mean(np.abs(z) < 3))

def summarize(results):
    """compare() for corv vs SPY and, as a baseline, SDSS pipeline vs SPY."""
    return {'corv': compare(results['rv'], results['e_rv'],
                            results['rv_spy'], results['e_rv_spy'])}

def frame_check(results, zmax = 5):
    """
    Slope of corv - SPY against each survey's heliocentric correction, with
    |z| > zmax outliers removed. A survey whose RVs were not heliocentric
    would give a slope of about +/-1; correct frames give about 0.

    Returns
    -------
    dict of survey: (slope, e_slope)
    """
    delta = results['rv'] - results['rv_spy']
    z = delta / np.hypot(results['e_rv'], results['e_rv_spy'])
    ok = np.isfinite(z) & (np.abs(z) < zmax)
    out = {}
    for survey in ('spy', 'sdss'):
        x = results['helio_' + survey]
        sel = ok & np.isfinite(x)
        pcoef, cov = np.polyfit(x[sel], delta[sel], 1, cov = True)
        out[survey] = (pcoef[0], np.sqrt(cov[0, 0]))
    return out

def plot(results, path):
    """RV comparison and histogram of error-normalized differences."""
    import matplotlib.pyplot as plt

    ok = np.isfinite(results['rv']) & np.isfinite(results['e_rv'])
    r = results[ok]
    z = (r['rv'] - r['rv_spy']) / np.hypot(r['e_rv'], r['e_rv_spy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 5))
    ax1.errorbar(r['rv_spy'], r['rv'], xerr = r['e_rv_spy'], yerr = r['e_rv'],
                 fmt = 'o', ms = 3, color = 'k', elinewidth = 0.8)
    lims = [min(r['rv_spy'].min(), r['rv'].min()) - 10, max(r['rv_spy'].max(), r['rv'].max()) + 10]
    ax1.plot(lims, lims, 'r--', lw = 1)
    ax1.set(xlim = lims, ylim = lims, xlabel = r'SPY $\langle RV \rangle$ (km/s)',
            ylabel = 'corv RV, SDSS (km/s)')

    bins = np.linspace(-6, 6, 25)
    ax2.hist(np.clip(z, -6, 6), bins = bins, density = True, color = '0.6')
    xx = np.linspace(-6, 6, 200)
    ax2.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'r', lw = 1)
    ax2.set(xlabel = r'(corv $-$ SPY) / $\sigma$', ylabel = 'density')

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description = __doc__.split('\n\n')[0])
    parser.add_argument('--cache', default = default_cache)
    parser.add_argument('--output', default = default_output)
    parser.add_argument('--model', default = '1d_da_nlte')
    parser.add_argument('--logp-min', type = float, default = -3)
    args = parser.parse_args()

    results = run(args.cache, args.model, logp_min = args.logp_min)
    os.makedirs(args.output, exist_ok = True)
    results.write(os.path.join(args.output, 'napiwotzki2020.ecsv'), overwrite = True)
    plot(results, os.path.join(args.output, 'napiwotzki2020.png'))

    print('\n%i spectra of %i stars' % (len(results), len(set(results['name']))))
    for label, stats in summarize(results).items():
        print('%-14s n = %i, offset = %.1f +/- %.1f km/s, scatter = %.1f km/s '
              '(median error %.1f), z scatter = %.2f, %.0f%% within 3 sigma' %
              (label, stats['n'], stats['median_offset'], stats['e_median_offset'],
               stats['scatter'], stats['median_error'], stats['z_scatter'],
               100 * stats['frac_within_3sigma']))
    for survey, (slope, e_slope) in frame_check(results).items():
        print('corv - SPY vs %s heliocentric correction: slope = %.2f +/- %.2f '
              '(0 if heliocentric, +/-1 if not)' % (survey.upper(), slope, e_slope))

if __name__ == '__main__':
    main()
