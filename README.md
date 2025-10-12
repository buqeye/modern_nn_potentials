# modern_nn_potentials

# Accompanying works

This repository contains all code and data necessary to generate the results in
*Assessing Correlated Truncation Errors in Modern Nucleon-Nucleon Potentials* (https://arxiv.org/abs/2402.13165)
and *Assessing Convergence Patterns Across Modern Nucleon-Nucleon Potentials* (https://arxiv.org/abs/2508.17558).

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
* To generate the figures found in *Assessing Convergence Patterns Across Modern Nucleon-Nucleon Potentials*, run `notebooks/figures2.ipynb` and
`notebooks/warping_playground.ipynb`.

Additionally, one can evaluate the posterior probability distribution functions for parameters of 
the theory such as the breakdown momentum scale $\Lambda_{b}$ and the soft scale $m_{\mathrm{eff}}$.
* To generate the values of these parameters found in *Assessing Correlated Truncation Errors in Modern Nucleon-Nucleon Potentials*'s Table III, run 
`scripts/posteriors_script_tab3_x.py` (with `x` replacing the number of the row of interest in that table).
* To generate the values of these parameters found in *Assessing Convergence Patterns Across Modern Nucleon-Nucleon Potentials*'s Table II, run 
`scripts/posteriors_script_warped_mpifixed.py` with the appropriate potential input in the code, and
`scripts/posteriors_script_scaledls_y.py` (with `y` replacing the short name of the potential of interest) for the variable $m_{\mathrm{eff}}$ results.

For best results, run on a computing cluster and not a local machine.

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
[arxiv]: https://arxiv.org/abs/2508.17558