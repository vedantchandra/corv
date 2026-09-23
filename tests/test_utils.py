import numpy as np
import pytest

import corv
from corv import utils
from corv.models import default_centres, default_windows, default_edges

def test_air_vac_roundtrip():
    air = np.linspace(3600, 9000, 100)
    vac = utils.air2vac(air)
    assert np.all(vac > air)
    assert np.allclose(utils.vac2air(vac), air, rtol = 0, atol = 1e-3)
    # H-alpha: 6562.80 (air) -> 6564.61 (vac)
    assert np.isclose(utils.air2vac(6562.80), 6564.61, atol = 0.01)

def test_doppler_shift():
    wl = np.linspace(4000, 7000, 30001)
    fl = 1 - 0.5 * np.exp(-0.5 * ((wl - 5500) / 5)**2)
    assert np.allclose(utils.doppler_shift(wl, fl, 0), fl)

    rv = 300
    shifted = utils.doppler_shift(wl, fl, rv)
    expected_centre = 5500 * np.sqrt((1 + rv / 2.99792458e5) / (1 - rv / 2.99792458e5))
    assert np.isclose(wl[np.argmin(shifted)], expected_centre, atol = 0.2)

def test_cont_norm_line_flattens_linear_continuum():
    wl = np.linspace(6400, 6700, 3000)
    continuum = 2 + 0.01 * (wl - 6400)
    line = 1 - 0.6 * np.exp(-0.5 * ((wl - 6564.61) / 10)**2)
    fl = continuum * line
    ivar = np.ones_like(wl)

    nwl, nfl, nivar = utils.cont_norm_line(wl, fl, ivar, 6564.61, 100, 25)
    assert nwl.min() >= 6464.61 and nwl.max() <= 6664.61
    assert np.allclose(nfl, line[(wl >= nwl[0]) & (wl <= nwl[-1])], atol = 1e-3)
    assert np.allclose(nivar * (fl[(wl >= nwl[0]) & (wl <= nwl[-1])] / nfl)**-2, 1)

def test_cont_norm_lines_concatenates(wl):
    fl = np.ones_like(wl)
    names = ['g', 'b']
    nwl, nfl, nivar = utils.cont_norm_lines(wl, fl, fl, names, default_centres,
                                            default_windows, default_edges)
    assert len(nwl) == len(nfl) == len(nivar)
    assert np.allclose(nfl, 1)
    assert nwl.min() < 4341.68 < 4862.68 < nwl.max()

def test_crrej_masks_cosmic_ray():
    rng = np.random.default_rng(0)
    wl = np.linspace(4000, 5000, 1000)
    fl = 1 + rng.normal(0, 0.01, len(wl))
    fl[500] = 5
    ivar = np.full_like(wl, 1e4)

    _, corr_fl, corr_ivar = utils.crrej(wl, fl, ivar.copy())
    assert corr_ivar[500] == 0
    assert abs(corr_fl[500] - 1) < 0.05
    assert (corr_ivar == 0).sum() < 10

# ---- regression tests from the audit ----

def test_crrej_does_not_modify_input():
    wl = np.linspace(4000, 5000, 200)
    fl = np.ones_like(wl)
    fl[100] = 5
    ivar = np.full_like(wl, 1e4)
    utils.crrej(wl, fl, ivar)
    assert np.all(ivar == 1e4)

def test_cont_norm_line_ignores_masked_edge_pixels():
    wl = np.linspace(6400, 6700, 3000)
    fl = np.ones_like(wl)
    ivar = np.ones_like(wl)
    edge_pixel = np.argmin(np.abs(wl - 6470))
    fl[edge_pixel - 5:edge_pixel + 5] = 50
    ivar[edge_pixel - 5:edge_pixel + 5] = 0

    _, nfl, nivar = utils.cont_norm_line(wl, fl, ivar, 6564.61, 100, 25)
    good = nivar > 0
    assert np.allclose(nfl[good], 1, atol = 1e-6)

def test_cont_norm_line_rejects_bad_edges():
    wl = np.linspace(6400, 6700, 3000)
    fl = np.ones_like(wl)
    with pytest.raises(ValueError):
        utils.cont_norm_line(wl, fl, fl, 6564.61, 100, 0)
    with pytest.raises(ValueError):
        utils.cont_norm_line(wl, fl, fl, 6564.61, 1, 25)

def test_cont_norm_line_masks_line_without_continuum():
    wl = np.linspace(6400, 6700, 3000)
    fl = np.ones_like(wl)
    with pytest.warns(UserWarning):
        _, nfl, nivar = utils.cont_norm_line(wl, fl, np.zeros_like(wl), 6564.61, 100, 25)
    assert np.all(nivar == 0) and np.all(np.isnan(nfl))

def test_lineplot_da_and_balmer_models(da_model, wl):
    params = da_model.model.make_params()
    fl = da_model.model.eval(params, x = wl)
    f = utils.lineplot(wl, fl, np.ones_like(wl), da_model.model, params)
    assert len(f.axes[0].texts) == 3

    balmer = corv.models.make_balmer_model(names = ['b', 'a'])
    f = utils.lineplot(wl, fl, np.ones_like(wl), balmer, balmer.make_params())
    assert len(f.axes[0].texts) == 1 # only chi2, no teff/logg

def test_plot_chi2_saves(tmp_path):
    rvgrid = np.linspace(-100, 100, 21)
    pcoef = np.array([0.01, 0, 5])
    path = tmp_path / 'chi2.png'
    utils.plot_chi2(rvgrid, np.polyval(pcoef, rvgrid), 0, 10, pcoef, path = str(path))
    assert path.exists()

def test_continuum_normalize_outputs():
    wl = np.linspace(4000, 5000, 500)
    fl = 1 + 0.001 * (wl - 4000)
    ivar = np.ones_like(wl)
    assert len(utils.continuum_normalize(wl, fl)) == 2
    _, fl_norm, ivar_norm, cont = utils.continuum_normalize(wl, fl, ivar, ret_cont = True)
    assert np.allclose(fl_norm * cont, fl) and np.allclose(ivar_norm, ivar * cont**2)
