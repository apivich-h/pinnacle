# PINNACLE: PINN Adaptive ColLocation and Experimental points selection

The code for paper titled "PINNACLE: PINN Adaptive ColLocation and Experimental points selection". The paper has been accepted as a spotlight paper at ICLR 2024 - refer to https://openreview.net/forum?id=GzNaCp6Vcg.

## Code structure

All the algorithm codes are kept in `pinnacle_code/deepxde_al_patch`. The code are based of `deepxde` package, patched in order to add in collocation point selection as required in our tests. The code has also been heavily adjusted to be compatible in Jax.

Example notebooks are in `pinnacle_code/*.ipynb`. The various notebooks contains examples to run our modules and methods to set up the training process.

Test scripts used for our experiment can be found in `pinnacle_code/al_pinn*.sh`. They are wrapper scripts with the test cases used, and they call other python scripts that does the experiment setups.

## Setup

Run `pip install -r requirements.txt` to install relevant packages. The scripts and notebooks can be used in the directory `pinnacle_code/` and will do imports accordingly.

Some physics simulation dataset need to be obtained from PDEBench. Refer to https://github.com/pdebench/PDEBench on how the dataset can be downloaded. Once installed their directory can be fed into the test scripts or to the test set loader. Our code includes a module which reads PDEBench data files so the `pdebench` package itself does not need to be installed.
