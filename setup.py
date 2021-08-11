from setuptools import setup

setup(name='corv',
      version='0.1',
      description='compact object radial velocities',
      author='Vedant Chandra',
      author_email='vedantchandra@g.harvard.edu',
      license='MIT',
      url='https://github.com/vedantchandra/corv',
      package_dir = {},
      packages=['corv'],
      package_data={'corv':['pkl/*']},
      dependency_links = [],
      install_requires=['numpy==1.18.5', 'scipy', 'lmfit'],
      include_package_data=True)
