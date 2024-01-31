# modern_nn_potentials

# Accompanying works

This repository contains all code and data necessary to generate the results in
*Fission fragment 1* (arXiv link here) and *Fission fragment 2* (arXiv link here).

## Installation

* This project relies on `python=3.8`. It was not tested with different versions.
  To view the entire list of required packages, see `environment.yml`.
* Note that a LaTeX distribution (such as Miktex) is also necessary in order to render the figures.
* Clone the repository to your local machine.
* Once you have `cd` into this repo, create a virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate modern-nn-potentials`
* Install the `cheftgp` package in the repo root directory using `pip install -e .`
  (you only need the `-e` option if you intend to edit the source code in `cheftgp/`).


## Contents

Beginning with order-by-order predictions of nucleon-nucleon (NN) scattering observables generated
using modern NN potentials that are implementations of chiral effective field theory ($\chi$EFT), 
this library extracts dimensionless coefficients according to the BUQEYE model.
* Data is found in `observables_data`.

From here, one can plot the resulting coefficients and test how well they are described as random 
draws from the same Gaussian process (GP) using statistical diagnostics.
* The operative script for this functionality is `scripts/diagnostics_script.py`.
* To generate the figures found in *FF1*, run `notebooks/figures1.ipynb`.
* To generate the figures found in *FF2*, run `notebooks/figures2.ipynb`.

Additionally, one can evaluate the posterior probability distribution functions for parameters of 
the theory such as the breakdown momentum scale $\Lambda_{b}$ and the soft scale $m_{\mathrm{eff}}$.
* The operative script for this functionality is `scripts/posteriors_script.py`.
* To generate the values of these parameters in *FF1*'s Table III, run 
`scripts/posteriors_script_tab3_x.py` (with `x` replacing the number of the row of interest).
* Accompanying data files are included in `scripts/data`.


## Miscellaneous

Unit tests can be found in `tests/unit_tests.ipynb`.

## Citing this work

Please cite this work as follows:

FF1 citation goes here.

[arxiv]: https://arxiv.org/abs/2301.05093
[Furnstahl]: https://www.sciencedirect.com/science/article/pii/S0370269320305220
[Drischler]: https://www.sciencedirect.com/science/article/pii/S0370269321007176
[Melendez]: https://www.sciencedirect.com/science/article/pii/S0370269321005487