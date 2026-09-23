import h5py
import numpy as np
import pytest

import corv

# ---- building grids from raw files ----

def write_raw_file(path, wavl, models):
    """Write a Koester-format file. models is a list of (teff, gravity, fnu)."""
    def block(values, fmt, per_line):
        return '\n'.join(' '.join(fmt % v for v in values[i:i + per_line])
                         for i in range(0, len(values), per_line))
    text = ' %i\n' % len(wavl) + block(wavl, '%10.2f', 10) + '\n'
    for teff, gravity, fnu in models:
        text += ' Effective temperature = %10.1f  gravity = %10.3E  y =  0.000E+00\n' % (teff, gravity)
        text += block(fnu, '%12.5E', 6) + '\n'
    path.write_text(text)

def fake_fnu(wavl, teff, logg):
    return (teff / 1e4) * (1 + logg / 10) * (1 + 1e-4 * wavl)

@pytest.fixture
def raw_grid(tmp_path):
    """
    Two-file synthetic grid: logg 7 on a fine wavelength grid, logg 8 on a
    coarse one, with the (teff=6000, logg=8) model missing.
    """
    fine = np.linspace(3000, 10000, 701)
    coarse = np.linspace(3000, 10000, 351)
    raw = tmp_path / 'raw' / 'test_grid'
    raw.mkdir(parents = True)
    write_raw_file(raw / '700', fine, [(t, 1e7, fake_fnu(fine, t, 7)) for t in (5000, 6000, 7000)])
    write_raw_file(raw / '800', coarse, [(t, 1e8, fake_fnu(coarse, t, 8)) for t in (5000, 7000)])
    return tmp_path / 'raw'

def build(build_model_grids, raw_dir, out, frame, monkeypatch):
    monkeypatch.setitem(build_model_grids.supported_models, 'test_grid', ('test_grid', frame))
    grid, frame, files = build_model_grids.build_grid('test_grid', str(raw_dir))
    build_model_grids.write_grid(str(out), 'test_grid', grid, frame, files, str(raw_dir))
    return grid

def test_read_model_file(build_model_grids, raw_grid):
    wavl, params, fluxes = build_model_grids.read_model_file(str(raw_grid / 'test_grid' / '700'))
    assert len(wavl) == 701 and np.isclose(wavl[0], 3000) and np.isclose(wavl[-1], 10000)
    assert [p[0] for p in params] == [5000, 6000, 7000]
    assert np.allclose([p[1] for p in params], 7)
    assert np.allclose(fluxes[1], fake_fnu(wavl, 6000, 7), rtol = 1e-5)

def test_read_model_file_rejects_truncated_spectrum(build_model_grids, tmp_path):
    wavl = np.linspace(4000, 5000, 20)
    write_raw_file(tmp_path / 'bad', wavl, [(5000, 1e8, np.ones(19))])
    with pytest.raises(AssertionError):
        build_model_grids.read_model_file(str(tmp_path / 'bad'))

@pytest.mark.parametrize('frame', ['vac', 'air'])
def test_build_and_load_grid(build_model_grids, raw_grid, tmp_path, frame, monkeypatch):
    out = tmp_path / 'grid.h5'
    build(build_model_grids, raw_grid, out, frame, monkeypatch)
    grid = corv.models.ModelGrid('test_grid', path = str(out))

    assert np.array_equal(grid.teff, [5000, 6000, 7000])
    assert np.allclose(grid.logg, [7, 8])
    assert grid.units == 'flam'

    # cropped to the fitting range, on the finer of the two raw grids
    raw_wavl = np.linspace(3000, 10000, 701)
    raw_wavl = raw_wavl[(3600 < raw_wavl) & (raw_wavl < 9000)]
    expected_wavl = corv.utils.air2vac(raw_wavl) if frame == 'air' else raw_wavl
    assert np.allclose(grid.wavl, expected_wavl)

    # fnu -> flam, evaluated at the raw (pre air->vac) wavelengths
    for i, teff in enumerate(grid.teff):
        for j, logg in enumerate(grid.logg):
            if not grid.valid[i, j]:
                continue
            expected = 2.99792458e18 * fake_fnu(raw_wavl, teff, logg) / raw_wavl**2
            assert np.allclose(grid.flux_grid[i, j], expected, rtol = 1e-4)

    # missing model is flagged
    assert grid.valid.sum() == 5
    assert not grid.valid[1, 1]
    assert np.all(grid.flux_grid[1, 1] == -999)

def test_rebuild_overwrites_group(build_model_grids, raw_grid, tmp_path, monkeypatch):
    out = tmp_path / 'grid.h5'
    with h5py.File(out, 'w') as f:
        f.create_group('other_grid')
    build(build_model_grids, raw_grid, out, 'vac', monkeypatch)
    build(build_model_grids, raw_grid, out, 'vac', monkeypatch)
    with h5py.File(out, 'r') as f:
        assert set(f.keys()) == {'other_grid', 'test_grid'}
        assert list(f['test_grid'].attrs['source_files']) == ['test_grid/700', 'test_grid/800']

def test_grid_model_uses_custom_grid(build_model_grids, raw_grid, tmp_path, monkeypatch):
    out = tmp_path / 'grid.h5'
    build(build_model_grids, raw_grid, out, 'vac', monkeypatch)
    model = corv.models.GridModel.from_hdf5('test_grid', grid_path = str(out))
    assert model.grid.modelname == 'test_grid'
    assert np.isfinite(model.grid.model_spec((5500, 7.5))).all()

# ---- the packaged grids ----

packaged_models = ['1d_da_nlte', '1d_elm_da_lte', '3d_da_lte_noh2', '3d_da_lte_h2', '3d_da_lte_old']

def test_packaged_file_contains_all_models():
    with h5py.File(corv.models.default_grid_path, 'r') as f:
        assert set(f.keys()) == set(packaged_models)

def test_unknown_model_raises():
    with pytest.raises(AssertionError):
        corv.models.ModelGrid('not_a_model')

@pytest.mark.parametrize('name', packaged_models)
def test_packaged_grid_is_consistent(name):
    grid = corv.models.ModelGrid(name)
    assert np.all(np.diff(grid.wavl) > 0)
    assert np.all(np.diff(grid.teff) > 0) and np.all(np.diff(grid.logg) > 0)
    assert 3600 < grid.wavl.min() and grid.wavl.max() < 9005
    assert grid.flux_grid.shape == (len(grid.teff), len(grid.logg), len(grid.wavl))
    assert np.array_equal(grid.valid, np.all(grid.flux_grid != -999, axis = -1))
    assert np.all(grid.flux_grid[grid.valid] > 0)

    # interpolator reproduces the grid at its nodes
    i, j = np.argwhere(grid.valid)[len(np.argwhere(grid.valid)) // 2]
    assert np.allclose(grid.model_spec((grid.teff[i], grid.logg[j])), grid.flux_grid[i, j])
