# modern_nn_potentials

# Caveat (Feburary 1, 2025)

The code herein is the subject of an ongoing review in preparation for the finalization and submission
of a new paper. Code appertaining to the already published Millican *et al.* (2024) paper (see below)
is in a worthy state to be downloaded and run, but this cannot be said for other code. Once the code 
for the forthcoming 2025 paper is finalized, it will be pushed here as version 1.0 and this caveat 
will be deleted.

# Accompanying works

This repository contains all code and data necessary to generate the results in
*Assessing Correlated Truncation Errors in Modern Nucleon-Nucleon Potentials* (https://arxiv.org/abs/2402.13165)
and another, follow-on paper to come in 2025.

## Installation

* This project relies on `python=3.8`. It was not tested with different versions.
  To view the entire list of required packages, see `environment.yml`.
* Note that a LaTeX distribution (such as Miktex) is also necessary in order to render the figures.
* Clone the repository to your local machine.
* Once you have `cd` into this repo, create a virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yaml
```
* Enter the virtual environment with `conda activate modern-nn-potentials`
* Install the `cheftgp` package in the repo root directory using `pip install -e .`
  (you only need the `-e` option if you intend to edit the source code in `cheftgp/`).


## Contents

Beginning with order-by-order predictions of nucleon-nucleon (NN) scattering observables generated
using modern NN potentials that are implementations of chiral effective field theory ($\chi$EFT), 
this library extracts dimensionless coefficients according to the BUQEYE model.
* Data for 12 potentials is found in `observables_data`.

From here, one can plot the resulting coefficients and test how well they are described as random 
draws from the same Gaussian process (GP) using statistical diagnostics.
* To generate the figures found in *Assessing Correlated Truncation Errors in Modern Nucleon-Nucleon Potentials*, run `notebooks/figures1.ipynb`.
* To generate the figures found in the forthcoming 2025 paper, run `notebooks/figures2.ipynb`.

Additionally, one can evaluate the posterior probability distribution functions for parameters of 
the theory such as the breakdown momentum scale $\Lambda_{b}$ and the soft scale $m_{\mathrm{eff}}$.
* To generate the values of these parameters in *Assessing Correlated Truncation Errors in Modern Nucleon-Nucleon Potentials*'s Table III, run 
`scripts/posteriors_script_tab3_x.py` (with `x` replacing the number of the row of interest in that table).
* Accompanying data files are included in `scripts/data`.

## Miscellaneous

Unit tests can be found in `tests/unit_tests.ipynb`.

## Citing this work

Please cite this work as follows:

```bibtex
@misc{millican2024assessing,
      title={Assessing Correlated Truncation Errors in Modern Nucleon-Nucleon Potentials}, 
      author={P. J. Millican and R. J. Furnstahl and J. A. Melendez and D. R. Phillips and M. T. Pratola},
      year={2024},
      eprint={2402.13165},
      archivePrefix={arXiv},
      primaryClass={nucl-th}
}
```

[arxiv]: https://arxiv.org/abs/2402.13165