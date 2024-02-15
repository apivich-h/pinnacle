import os
import sys
import pickle as pkl
from functools import partial
import random
import argparse
from datetime import datetime
import traceback

import numpy as np
import tqdm

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
from IPython.display import HTML

plt.rcParams['figure.figsize'] = (6, 5)
plt.rcParams['figure.dpi'] = 300

plt.rcParams.update({
    'font.size': 15,
    'text.usetex': False,
})

import jax
import jax.numpy as jnp

os.environ["DDE_BACKEND"] = "jax"
from deepxde_al_patch import deepxde as dde

from deepxde_al_patch.model_loader import construct_model
from deepxde_al_patch.utils import get_pde_residue
from deepxde_al_patch.modified_train_loop import ModifiedTrainLoop
from deepxde_al_patch.plotters import plot_residue_loss, plot_error, plot_prediction, plot_eigvals, plot_eigenbasis


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# only run JAX on cpu
# jax.config.update('jax_platform_name', 'cpu')

# set precision for jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

try:
    print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs={jax.local_device_count("gpu")}')
except RuntimeError:
    print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs=None')


parser = argparse.ArgumentParser()

parser.add_argument('--results_dir', type=str, default='al_pinn_results_const_change')
parser.add_argument('--saved_models_dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models'))
parser.add_argument('--pdebench_dir', type=str, default='~/pdebench')

parser.add_argument('--eqn', type=str)  # heat, diff
parser.add_argument('--init_const', type=float, nargs="+")
parser.add_argument('--use_pdebench', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--allow_ic', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--inverse', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--param_ver', type=int, default=0)

parser.add_argument('--nn_arch', type=str, default=None)  # normal, fourier
parser.add_argument('--hidden_layers', type=int, default=2)  # normal, fourier
parser.add_argument('--hidden_dim', type=int, default=128)  # normal, fourier
parser.add_argument('--optim', type=str, default='adam')

parser.add_argument('--train_steps', type=int, default=50000)
parser.add_argument('--al_every', type=int, default=5000)
parser.add_argument('--select_anchors_every', type=int, default=5000)
parser.add_argument('--loss_w_bcs', type=float, default=1.0)
parser.add_argument('--autoscale_loss_w_bcs', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--auto_al', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--anchor_budget', type=int, default=0)

args = parser.parse_args()

print(args)

PDEBENCH_DATA = os.path.expanduser(args.pdebench_dir)
RESULTS_FOLDER = os.path.expanduser(args.results_dir)
saved_models_dir = os.path.expanduser(args.saved_models_dir)

eqn = args.eqn
init_const = args.init_const
nn_arch = args.nn_arch
hidden_layers = args.hidden_layers
hidden_dim = args.hidden_dim
optim = args.optim
use_pdebench = args.use_pdebench
data_seed = args.data_seed
inverse = args.inverse
param_ver = args.param_ver

num_domain = 500
num_icbc = 300


print(f"""===== ARGUMENTS =====
eqn = {eqn}
init_const = {init_const}
data_seed = {data_seed}
param_ver = {param_ver}
inverse = {inverse}

nn_arch = {nn_arch}
hidden_layers = {hidden_layers}
hidden_dim = {hidden_dim}
optim = {optim}""")

print(f'=====================\n')

data_code = f'{eqn}-{"-".join(str(c) for c in init_const)}'
model_code = f'{nn_arch}-{hidden_layers}-{hidden_dim}'
param_prefix = f'{data_code}_pb-{data_seed}_{model_code}_v{param_ver}'

"""TRAIN INIT MODEL IF NEEDED """

model_path = os.path.join(saved_models_dir, f'{param_prefix}.pkl')

if not os.path.isfile(model_path):
    
    print(f'{model_path} does not exist, need to do training.')
    max_steps = 50000
    
    model, _ = construct_model(
        pde_name=eqn, 
        pde_const=init_const, 
        use_pdebench=use_pdebench,
        data_seed=data_seed,
        inverse_problem=False,
        data_root=PDEBENCH_DATA,
        num_domain=1000, 
        num_boundary=200, 
        num_initial=200,
        include_ic=True,
        hidden_layers=hidden_layers, 
        hidden_dim=hidden_dim, 
        activation='tanh', 
        initializer='Glorot uniform', 
        arch=nn_arch, 
    )
    
    train_loop = ModifiedTrainLoop(
        model=model, 
        point_selector_method='random',
        point_selector_args=dict(res_proportion=0.8),
        train_steps=max_steps,
        al_every=50000,
        select_anchors_every=50000,
        mem_pts_total_budget=10000,
        anchor_budget=1000,
        snapshot_every=5000,
        optim_method='adam', 
        optim_lr=1e-4, 
        optim_args=dict(),
    )
    
    train_loop.train()
    
    with open(model_path, 'wb+') as f:
        pkl.dump(train_loop.snapshot_data[max_steps]['params'], f)
        
else:
    print(f'{model_path} exists, will use pickled params.')