import importlib.util
import os

import numpy as np
import pytest
from astropy.table import Table

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope = 'module')
def napiwotzki2020():
    """The validation/napiwotzki2020.py module."""
    spec = importlib.util.spec_from_file_location(
        'napiwotzki2020', os.path.join(repo, 'validation', 'napiwotzki2020.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ---- offline checks of the validation logic ----

def test_select_constant(napiwotzki2020):
    stars = Table(dict(name = ['ok', 'binary', 'maybe', 'single_epoch', 'dd', 'magnetic', 'dd_fit'],
                       logp = [-0.5, -6, -3.5, np.nan, -1, -1, -1],
                       n_logp = ['', 'DD', '', '', 'dd', '', ''],
                       remark = ['GBD12,phot', '', '', '', '', 'magnetic', 'primary, WB94']))
    assert list(napiwotzki2020.select_constant(stars)['name']) == ['ok']
    assert list(napiwotzki2020.select_constant(stars, logp_min = -4)['name']) == ['ok', 'maybe']

def test_compare(napiwotzki2020):
    rng = np.random.default_rng(0)
    rv_ref = rng.uniform(-50, 100, 2000)
    rv = rv_ref + 5 + rng.normal(0, 3, rv_ref.size)
    e = np.full(rv_ref.size, 3 / np.sqrt(2))
    rv[0] = np.nan # ignored
    stats = napiwotzki2020.compare(rv, e, rv_ref, e)
    assert stats['n'] == 1999
    assert np.isclose(stats['median_offset'], 5, atol = 4 * stats['e_median_offset'])
    assert np.isclose(stats['scatter'], 3, rtol = 0.1)
    assert np.isclose(stats['z_scatter'], 1, rtol = 0.1)

def test_load_spy_fills_cached_strings(napiwotzki2020, tmp_path):
    Table(dict(name = ['a', 'b'], n_logp = ['', 'DD'], remark = ['', 'magnetic'],
               helio_spy = [1., 2.])).write(tmp_path / 'spy_stars.ecsv')
    stars = napiwotzki2020.load_spy(str(tmp_path))
    assert list(stars['n_logp']) == ['', 'DD'] and list(stars['remark']) == ['', 'magnetic']

def test_frame_check_detects_uncorrected_frame(napiwotzki2020):
    rng = np.random.default_rng(1)
    n = 200
    results = Table(dict(rv_spy = rng.uniform(-50, 100, n), e_rv = np.full(n, 5.),
                         e_rv_spy = np.full(n, 2.), helio_spy = rng.uniform(-30, 30, n),
                         helio_sdss = rng.uniform(-30, 30, n)))
    # SDSS left uncorrected: its velocities are off by -helio_sdss
    results['rv'] = results['rv_spy'] - results['helio_sdss'] + rng.normal(0, 5, n)
    slopes = napiwotzki2020.frame_check(results, zmax = np.inf)
    assert abs(slopes['spy'][0]) < 3 * slopes['spy'][1]
    assert np.isclose(slopes['sdss'][0], -1, atol = 3 * slopes['sdss'][1])

def test_spy_heliocentric_correction_varies_over_year(napiwotzki2020):
    stars = Table(dict(name = ['x'], ra = [0.], dec = [0.]))
    corr = [napiwotzki2020.spy_heliocentric_correction(
                stars, Table(dict(name = ['x'], hjd = [jd], e_hrv = [1.])))[0]
            for jd in (2451545.0, 2451545.0 + 182.6)]
    # Earth's orbital velocity (~30 km/s) projected on a star at the ecliptic plane
    assert all(abs(c) < 31 for c in corr) and abs(corr[0] - corr[1]) > 40

# ---- the validation itself (network, slow) ----

# Regression limits, set with margin from the first full run (122 SDSS spectra
# of 75 stars): offset -3.3 +/- 1.1 km/s, z scatter 1.47, 91% within 3 sigma.
MAX_OFFSET = 8           # km/s
MAX_Z_SCATTER = 2
MIN_FRAC_3SIGMA = 0.8
MAX_FRAME_SLOPE = 0.3     # measured -0.10 +/- 0.08 (SPY), -0.09 +/- 0.09 (SDSS)

@pytest.mark.validation
def test_rvs_consistent_with_napiwotzki2020(napiwotzki2020):
    results = napiwotzki2020.run(verbose = False)
    stats = napiwotzki2020.summarize(results)['corv']
    assert stats['n'] >= 0.9 * len(results) # nearly all fits succeed
    assert abs(stats['median_offset']) < MAX_OFFSET
    assert stats['z_scatter'] < MAX_Z_SCATTER
    assert stats['frac_within_3sigma'] > MIN_FRAC_3SIGMA
    # both surveys heliocentric: a wrong frame would give a slope of about +/-1
    for survey, (slope, e_slope) in napiwotzki2020.frame_check(results).items():
        assert abs(slope) < MAX_FRAME_SLOPE, survey
