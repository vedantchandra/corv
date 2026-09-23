import numpy as np
import pytest

import corv

true_rv = 123.

@pytest.fixture(scope = 'module')
def mock_spectrum(da_model):
    """Noisy DA spectrum at teff=13000, logg=8.1, RV=123 km/s, S/N ~ 50."""
    wl = np.linspace(3800, 7000, 4000)
    params = da_model.model.make_params()
    params['teff'].set(value = 13000)
    params['logg'].set(value = 8.1)
    params['RV'].set(value = true_rv)
    fl = da_model.model.eval(params, x = wl)

    rng = np.random.default_rng(42)
    sigma = 0.02 * np.ones_like(wl)
    fl = fl + rng.normal(0, sigma)
    return wl, fl, 1 / sigma**2

def test_normalized_residual_zero_for_perfect_model(da_model):
    wl = np.linspace(3800, 7000, 4000)
    params = da_model.model.make_params()
    fl = da_model.model.eval(params, x = wl)
    resid = corv.fit.normalized_residual(wl, fl, np.ones_like(wl), da_model.model, params)
    assert np.allclose(resid, 0, atol = 1e-8)

def test_fit_rv_recovers_rv(da_model, mock_spectrum):
    wl, fl, ivar = mock_spectrum
    params = da_model.model.make_params()
    params['teff'].set(value = 13000)
    params['logg'].set(value = 8.1)

    rv, e_rv, redchi, rvgrid, cc = corv.fit.fit_rv(wl, fl, ivar, da_model.model, params,
                                                   min_rv = -500, max_rv = 500, npoints = 201)
    assert 0 < e_rv < 20
    assert abs(rv - true_rv) < 4 * e_rv
    assert 0.5 < redchi < 2
    assert len(rvgrid) == len(cc)

def test_fit_rv_saves_plot(da_model, mock_spectrum, tmp_path):
    wl, fl, ivar = mock_spectrum
    params = da_model.model.make_params()
    path = tmp_path / 'xcorr.png'
    corv.fit.fit_rv(wl, fl, ivar, da_model.model, params, min_rv = -500, max_rv = 500,
                    npoints = 51, plot = True, path = str(path))
    assert path.exists()

def test_fit_corv_recovers_parameters(da_model, mock_spectrum):
    wl, fl, ivar = mock_spectrum
    rv, e_rv, redchi, param_res = corv.fit.fit_corv(wl, fl, ivar, da_model.model,
                                                    rv_kw = dict(min_rv = -500, max_rv = 500,
                                                                    npoints = 201))
    assert abs(rv - true_rv) < 4 * e_rv
    assert abs(param_res.params['teff'].value - 13000) < 500
    assert abs(param_res.params['logg'].value - 8.1) < 0.1

# ---- regression tests from the audit ----

def test_fit_rv_does_not_modify_params(da_model, mock_spectrum):
    wl, fl, ivar = mock_spectrum
    params = da_model.model.make_params()
    params['RV'].set(value = 42)
    corv.fit.fit_rv(wl, fl, ivar, da_model.model, params, min_rv = -500, max_rv = 500, npoints = 21)
    assert params['RV'].value == 42

def test_fit_rv_redchi_ignores_masked_pixels(da_model, mock_spectrum):
    wl, fl, ivar = mock_spectrum
    params = da_model.model.make_params()
    params['teff'].set(value = 13000)
    params['logg'].set(value = 8.1)
    kw = dict(min_rv = -500, max_rv = 500, npoints = 101)
    _, _, redchi, _, _ = corv.fit.fit_rv(wl, fl, ivar, da_model.model, params, **kw)

    masked = ivar.copy()
    masked[::2] = 0
    _, _, redchi_masked, _, _ = corv.fit.fit_rv(wl, fl, masked, da_model.model, params, **kw)
    assert np.isclose(redchi_masked, redchi, rtol = 0.2)

def test_fit_rv_flat_chi2_does_not_raise(da_model):
    """A featureless spectrum gives no chi2 minimum; this should warn, not crash."""
    wl = np.linspace(3800, 7000, 4000)
    fl = np.ones_like(wl)
    params = da_model.model.make_params()
    params['teff'].set(value = 13000)
    with pytest.warns(UserWarning):
        rv, e_rv, redchi, rvgrid, cc = corv.fit.fit_rv(wl, fl, np.zeros_like(wl), da_model.model,
                                                       params, min_rv = -500, max_rv = 500,
                                                       npoints = 21)
    assert np.isfinite(rv) and not np.isfinite(e_rv)

def test_fit_params_keeps_best_start(da_model, mock_spectrum):
    wl, fl, ivar = mock_spectrum
    single = [corv.fit.fit_params(wl, fl, ivar, da_model.model, init_teffs = [t])
              for t in (8000, 13000)]
    best = corv.fit.fit_params(wl, fl, ivar, da_model.model, init_teffs = [8000, 13000])
    assert np.isclose(best.redchi, min(r.redchi for r in single))

def test_fit_corv_sets_rv_on_params(da_model, mock_spectrum):
    wl, fl, ivar = mock_spectrum
    rv, e_rv, _, param_res = corv.fit.fit_corv(wl, fl, ivar, da_model.model,
                                               xcorr_kw = dict(min_rv = -500, max_rv = 500,
                                                               npoints = 51))
    assert param_res.params['RV'].value == rv and param_res.params['RV'].stderr == e_rv

def test_xcorr_rv_alias():
    assert corv.fit.xcorr_rv is corv.fit.fit_rv

def test_fit_corv_balmer_model(mock_spectrum):
    wl, fl, ivar = mock_spectrum
    corvmodel = corv.models.make_balmer_model(names = ['d', 'g', 'b', 'a'])
    rv, e_rv, _, _ = corv.fit.fit_corv(wl, fl, ivar, corvmodel, init_teffs = [10000, 20000],
                                       rv_kw = dict(min_rv = -500, max_rv = 500, npoints = 101))
    assert abs(rv - true_rv) < 50

def test_fit_corv_with_masked_line(da_model, mock_spectrum):
    """A line with no usable continuum is masked (NaN flux, zero ivar) and must not break the fit."""
    wl, fl, ivar = mock_spectrum
    ivar = ivar.copy()
    ivar[np.abs(wl - 6564.61) < 101] = 0
    with pytest.warns(UserWarning, match = 'no usable continuum'):
        rv, e_rv, _, param_res = corv.fit.fit_corv(wl, fl, ivar, da_model.model,
                                                   rv_kw = dict(min_rv = -500, max_rv = 500,
                                                                npoints = 101))
    assert abs(rv - true_rv) < 4 * e_rv
    assert abs(param_res.params['teff'].value - 13000) < 1000
