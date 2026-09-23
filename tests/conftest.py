import importlib.util
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import corv

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def pytest_addoption(parser):
    parser.addoption('--run-validation', action = 'store_true',
                     help = 'run the slow validation tests, which download external data')

def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-validation'):
        return
    skip = pytest.mark.skip(reason = 'needs --run-validation')
    for item in items:
        if 'validation' in item.keywords:
            item.add_marker(skip)

@pytest.fixture(scope='session')
def build_model_grids():
    """The scripts/build_model_grids.py module."""
    spec = importlib.util.spec_from_file_location(
        'build_model_grids', os.path.join(repo, 'scripts', 'build_model_grids.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope='session')
def wl():
    return np.linspace(3700, 7000, 6000)

@pytest.fixture(scope='session')
def da_model():
    return corv.models.GridModel.from_hdf5('1d_da_nlte', names = ['d', 'g', 'b', 'a'])
