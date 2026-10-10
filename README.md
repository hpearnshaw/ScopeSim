# UVEX ScopeSim
## A telescope observation simulator for Python

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Summary

This instrument simulator for UVEX is based on a modified UVEX fork of AstarVienna's 
ScopeSim package, containing custom Effects and adjustments to support a UVEX instrument 
simulator. Changes include:
- Field-varying and wavelength-dependent PSFs
- Custom vignetting models
- UVEX configuration information incorporated directly into the simulator codebase
- A utility module for converting various inputs into ScopeSim-compatible Source objects

Coming soon:
- Field distortion and projection effects for wide-field optics
- 2D per-pixel maps for detector effects

This package is currently under development and should not be considered indicative of 
UVEX performance at this time.

For more information about ScopeSim itself, see below and source repository here: [ScopeSim](https://github.com/AstarVienna/ScopeSim)

## ScopeSim

ScopeSim aims to simulate images of astronomical objects observed with visual
and infrared instruments. It does this by creating models of the optical train
and astronomical objects and then pushing the object through the optical train.
The resulting 2D image is then broadcast to a detector chip and read out into a
FITS file.

This code was originally based on the [SimCADO](https://github.com/astronomyk/simcado) package

## Documentation
The main set of ScopeSim documentation can be found here:
https://scopesim.readthedocs.io/en/latest/

A basic Jupyter Notebook can be found here:
[scopesim_basic_intro.ipynb](docs/source/examples/1_scopesim_intro.ipynb)

### Feature roadmap
The [ScopeSim Feature Roadmap](https://github.com/orgs/AstarVienna/projects/21) shows
what AstarVienna are currently working on. 
