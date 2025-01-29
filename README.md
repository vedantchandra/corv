# corv: Compact Object Radial Velocities

this package contains routines to model the spectra of compact objects (e.g. white dwarfs and M-dwarfs), and estimate their radial velocities. 

To install, run: ``pip install git+https://github.com/vedantchandra/corv``.

The following model white dwarf spectra can be used to fit RVs:
* `1d_da_nlte` : [1D pure-hydrogen (DA) non-local thermodynamic equilibrium (NLTE) spectra](https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/readme.txt)
* `1d_elm_da_lte` : [1D pure-hydrogen (DA) extremely low-mass (ELM) LTE spectra](https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/readme_elm.txt)
* `3d_da_lte_noh2` [⟨3D⟩ pure-hydrogen (DA) LTE spectra without molecular H2 lines](https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/readme_3d.txt)
* `3d_da_lte_h2` [⟨3D⟩ pure-hydrogen (DA) LTE spectra with molecular H2 lines](https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/readme_3d.txt)
* `3d_da_lte_old` Archived 3D DA LTE spectra (these were previously the `corv` defaults but are no longer recommended)

Note that `corv` does not apply 3D corrections to 1D parameters by default. These should be done using the routines available in the model grid documentation.

*Update (01/29/2025):* I've added an optional term to enable fitting Voigt profiles with skewness as a free parameter! The accuracy of this is not yet tested, but if it works it may enable more accurate treatment of line asymmetries due to higher order Stark effects. --Stefan

## Contributors

[Vedant Chandra](https://vedantchandra.com/) (Harvard)

[Stefan Arseneau](https://stefanarseneau.github.io) (Boston University)

[Keith P. Inight](https://warwick.ac.uk/fac/sci/physics/research/astro/people/keithinight/) (Warwick)
