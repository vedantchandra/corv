from setuptools import setup,find_packages

setup(name='corv',
	version='0.1',
	description='compact object radial velocities',
	author='Vedant Chandra',
	author_email='vedant.chandra@cfa.harvard.edu',
	license='MIT',
	url='https://github.com/vedantchandra/corv',
	package_dir = {"" : "src"},
	packages=find_packages(where='src'),
	package_data={'corv':['models/*']},
	dependency_links = [],
	install_requires=['numpy', 'scipy', 'lmfit', 'matplotlib', 'astropy', 'tqdm'],
	include_package_data=True)