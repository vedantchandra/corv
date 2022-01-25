from setuptools import setup,find_packages

setup(name='corv',
      version='0.1',
      description='compact object radial velocities',
      author='Vedant Chandra',
      author_email='vedantchandra@g.harvard.edu',
      license='MIT',
      url='https://github.com/vedantchandra/corv',
      package_dir = {"" : "src"},
      packages=find_packages(where='src'),
      package_data={'corv':['pkl/*']},
      dependency_links = [],
      install_requires=['numpy', 'scipy', 'lmfit', 'matplotlib', 'astropy'],
      include_package_data=True)
