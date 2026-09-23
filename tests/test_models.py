import numpy as np
import pytest

import corv

c_kms = 2.99792458e5

def line_minimum(wl, fl, centre, half_width = 20):
    sel = np.abs(wl - centre) < half_width
    return wl[sel][np.argmin(fl[sel])]

def test_balmer_model_normalized_and_shifts(wl):
    corvmodel = corv.models.make_balmer_model(names = ['a'])
    params = corvmodel.make_params()

    nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
    assert len(nwl) == len(nfl) > 0
    assert np.all(np.abs(nwl - 6564.61) <= 100)
    assert np.isfinite(nfl).all()

    rv = 1000
    params['RV'].set(value = rv)
    _, nfl_shifted = corv.models.get_normalized_model(wl, corvmodel, params)
    expected = 6564.61 / np.sqrt((1 - rv / c_kms) / (1 + rv / c_kms))
    assert np.isclose(line_minimum(nwl, nfl_shifted, expected), expected, atol = 1)

def test_skewed_balmer_model_has_skew_params():
    corvmodel = corv.models.make_balmer_model(names = ['a', 'b'], nvoigt = 2, skewness = True)
    params = corvmodel.make_params()
    assert {'a0_skew', 'a1_skew', 'b0_skew', 'b1_skew', 'RV'} <= set(params)
    # secondary components are tied to the primary centre
    assert params['a1_center'].expr == 'a0_center'

@pytest.mark.parametrize('name', ['1d_da_nlte', '1d_elm_da_lte', '3d_da_lte_noh2',
                                  '3d_da_lte_h2', '3d_da_lte_old'])
def test_grid_model_evaluates(name, wl):
    corvmodel = corv.models.GridModel.from_hdf5(name).model
    params = corvmodel.make_params()
    params['teff'].set(value = 12000)
    params['logg'].set(value = 8)

    fl = corvmodel.eval(params, x = wl)
    assert np.isfinite(fl).all()
    assert np.isclose(np.median(fl), 1, atol = 0.1)

    nwl, nfl = corv.models.get_normalized_model(wl, corvmodel, params)
    assert np.isfinite(nfl).all()
    # H-beta core is well below the continuum at 12000 K
    assert nfl[np.argmin(np.abs(nwl - 4862.68))] < 0.7

def test_grid_model_rv_shift(da_model, wl):
    params = da_model.model.make_params()
    rest = da_model.model.eval(params, x = wl)
    rv = 500
    params['RV'].set(value = rv)
    shifted = da_model.model.eval(params, x = wl)

    for centre in (4862.68, 6564.61):
        expected = centre / np.sqrt((1 - rv / c_kms) / (1 + rv / c_kms))
        assert np.isclose(line_minimum(wl, rest, centre), centre, atol = 1)
        assert np.isclose(line_minimum(wl, shifted, expected), expected, atol = 1)

def test_grid_model_param_bounds():
    params = corv.models.GridModel.from_hdf5('3d_da_lte_h2').model.make_params()
    assert params['teff'].min == 4001 and params['teff'].max == 40000
    assert np.isclose(params['logg'].min, 7, atol = 1e-3) and np.isclose(params['logg'].max, 9, atol = 1e-3)
    assert params['res'].vary is False

    params = corv.models.GridModel.from_hdf5('1d_da_nlte').model.make_params()
    assert np.isclose(params['logg'].max, 9.5, atol = 1e-3)

# ---- regression tests from the audit ----

def applied_sigma(corvmodel, dx, res):
    """Gaussian sigma (AA) that the model's resolution kernel actually applies."""
    import scipy.ndimage as nd
    wl = np.arange(4000, 5600, dx)
    params = corvmodel.make_params()
    params['res'].set(value = 1e-9)
    raw = corvmodel.eval(params, x = wl)
    params['res'].set(value = res)
    out = corvmodel.eval(params, x = wl)
    sigmas = np.arange(0.1, 12, 0.05)
    return sigmas[np.argmin([np.sum((nd.gaussian_filter1d(raw, s / dx) - out)**2) for s in sigmas])]

@pytest.mark.parametrize('dx', [0.5, 2.0])
def test_resolution_is_sigma_in_angstrom(da_model, dx):
    assert np.isclose(applied_sigma(da_model.model, dx, 2.0), 2.0, atol = 0.1)

@pytest.mark.parametrize('dx', [0.5, 2.0])
def test_generic_resolution_is_sigma_in_angstrom(dx):
    grid = corv.models.ModelGrid('1d_da_nlte')
    corvmodel = corv.models.GridModel(grid.model_spec, grid.wavl).model
    assert np.isclose(applied_sigma(corvmodel, dx, 2.0), 2.0, atol = 0.1)

def test_generic_model_has_no_stray_params():
    grid = corv.models.ModelGrid('1d_da_nlte')
    params = corv.models.GridModel(grid.model_spec, grid.wavl).model.make_params()
    assert set(params) == {'teff', 'logg', 'RV', 'res'}

@pytest.mark.parametrize('name', ['1d_da_nlte', '1d_elm_da_lte', '3d_da_lte_noh2',
                                  '3d_da_lte_h2', '3d_da_lte_old'])
def test_grid_bounds_inside_grid(name):
    model = corv.models.GridModel.from_hdf5(name)
    grid, params = model.grid, model.model.make_params()
    assert grid.teff.min() <= params['teff'].min < params['teff'].max <= grid.teff.max()
    assert grid.logg.min() <= params['logg'].min < params['logg'].max <= grid.logg.max()
    # the model evaluates at the bound corners
    for teff in (params['teff'].min, params['teff'].max):
        for logg in (params['logg'].min, params['logg'].max):
            assert np.isfinite(grid.model_spec((teff, logg))).all()

def test_elm_bounds_reach_low_gravity():
    params = corv.models.GridModel.from_hdf5('1d_elm_da_lte').model.make_params()
    assert params['logg'].min < 6

def test_grid_bounds_can_be_overridden():
    params = corv.models.GridModel.from_hdf5('1d_da_nlte', teff_bounds = (6000, 30000),
                                                 logg_bounds = (7.5, 8.5)).model.make_params()
    assert (params['teff'].min, params['teff'].max) == (6000, 30000)
    assert (params['logg'].min, params['logg'].max) == (7.5, 8.5)

def test_generic_bounds_from_interpolator_grid():
    grid = corv.models.ModelGrid('1d_da_nlte')
    params = corv.models.GridModel(grid.model_spec, grid.wavl).model.make_params()
    assert (params['teff'].min, params['teff'].max) == (grid.teff.min(), grid.teff.max())
    assert (params['logg'].min, params['logg'].max) == (grid.logg.min(), grid.logg.max())

def test_generic_interpolator_without_grid_needs_bounds():
    grid = corv.models.ModelGrid('1d_da_nlte')
    interp = lambda p: grid.model_spec(p)
    with pytest.raises(AssertionError):
        corv.models.GridModel(interp, grid.wavl)
    params = corv.models.GridModel(interp, grid.wavl, (6000, 30000), (7.5, 8.5)).model.make_params()
    assert (params['teff'].min, params['logg'].max) == (6000, 8.5)

def test_backwards_compatible_names():
    model = corv.models.WarwickDAModel('1d_da_nlte', names = ['a'])
    assert isinstance(model, corv.models.GridModel) and model.model.names == ['a']

def test_koester_missing_package_raises(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'koester', None)
    with pytest.raises(ImportError, match = 'koester'):
        corv.models.GridModel.from_koester()
    with pytest.raises(ImportError, match = 'koester'):
        corv.models.make_koester_model()
