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

parser.add_argument('--results_dir', type=str, default='al_pinn_results_ic_change')
parser.add_argument('--saved_models_dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models'))
parser.add_argument('--pdebench_dir', type=str, default='~/pdebench')

parser.add_argument('--eqn', type=str)  # heat, diff
parser.add_argument('--const', type=float, nargs="+")  # equation constants
# parser.add_argument('--use_pdebench', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--data_seed_pair', type=int, nargs="+")
parser.add_argument('--allow_ic', action=argparse.BooleanOptionalAction, default=True)
# parser.add_argument('--inverse', action=argparse.BooleanOptionalAction, default=False)
# parser.add_argument('--inverse_guess', type=float, nargs="+", default=None)  # equation constants
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

parser.add_argument('--method', type=str, default='random')
parser.add_argument('--num_points', type=int, default=50)
parser.add_argument('--mem_pts_total_budget', type=int, default=1000)

parser.add_argument('--rand_method', type=str, default='pseudo')
parser.add_argument('--rand_res_prop', type=float, default=0.8)

parser.add_argument('--res_res_prop', type=float, default=0.8)
parser.add_argument('--res_all_types', action=argparse.BooleanOptionalAction, default=False)

parser.add_argument('--eig_weight_method', type=str, default='labels')
parser.add_argument('--eig_memory', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--eig_sampling', type=str, default='pseudo')
parser.add_argument('--eig_scale', type=str, default='none')

parser.add_argument('--gd_indicator', type=str, default='K')
parser.add_argument('--gd_compare_mode', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--gd_crit', type=str, default='fr')

args = parser.parse_args()

print(args)

PDEBENCH_DATA = os.path.expanduser(args.pdebench_dir)
RESULTS_FOLDER = os.path.expanduser(args.results_dir)
saved_models_dir = os.path.expanduser(args.saved_models_dir)

eqn = args.eqn
const = args.const
nn_arch = args.nn_arch
hidden_layers = args.hidden_layers
hidden_dim = args.hidden_dim
optim = args.optim
# use_pdebench = args.use_pdebench
data_seed_pair = args.data_seed_pair
# inverse = args.inverse
# inverse_guess = args.inverse_guess
param_ver = args.param_ver
allow_ic = args.allow_ic

train_steps = args.train_steps
al_every = args.al_every
select_anchors_every = args.select_anchors_every
mem_pts_total_budget = args.mem_pts_total_budget
anchor_budget = args.anchor_budget
loss_w_bcs = args.loss_w_bcs
autoscale_loss_w_bcs = args.autoscale_loss_w_bcs
auto_al = args.auto_al

method = args.method
num_points = args.num_points

rand_method = args.rand_method
rand_res_prop = args.rand_res_prop

res_res_prop = args.res_res_prop
res_all_types = args.res_all_types

eig_weight_method = args.eig_weight_method
eig_memory = args.eig_memory
eig_sampling = args.eig_sampling
eig_scale = args.eig_scale

gd_indicator = args.gd_indicator
gd_compare_mode = args.gd_compare_mode
gd_crit = args.gd_crit

num_domain = 500
num_icbc = 300

assert len(data_seed_pair) == 2
init_data_seed, tuned_seed_pair = data_seed_pair

print(f"""===== ARGUMENTS =====
eqn = {eqn}
const = {const}
init_data_seed = {init_data_seed}
tuned_seed_pair = {tuned_seed_pair}
param_ver = {param_ver}
allow_ic = {allow_ic}

nn_arch = {nn_arch}
hidden_layers = {hidden_layers}
hidden_dim = {hidden_dim}
optim = {optim}

train_steps = {train_steps}
mem_pts_total_budget = {mem_pts_total_budget}
anchor_budget = {anchor_budget}
al_every = {al_every}
select_anchors_every = {select_anchors_every}
loss_w_bcs = {loss_w_bcs}
autoscale_loss_w_bcs = {autoscale_loss_w_bcs}
auto_al = {auto_al}

method = {method}
num_points = {num_points}""")

if method == 'random':
    method_str = f'random_{rand_method}_prop-{rand_res_prop}'
    print(f"""
rand_method = {rand_method}
rand_res_prop = {rand_res_prop}""")
    
elif method == 'residue':
    method_str = f'residue_prop-{res_res_prop}' + ('_alltype' if res_all_types else '')
    print(f"""
res_res_prop = {res_res_prop}
res_all_types = {res_all_types}""")

elif method in {'greedy', 'kmeans', 'sampling'}:
    method_str = f'{method}_{eig_weight_method}_scale-{eig_scale}' + ('_mem' if eig_memory else '')
    print(f"""
eig_weight_method = {eig_weight_method}
eig_memory = {eig_memory}
eig_sampling = {eig_sampling}
eig_scale = {eig_scale}""")
    
elif method == 'gd':
    method_str = f'gd_{gd_indicator}_{gd_crit}' + ('_fulldiff' if gd_compare_mode else '')
    print(f"""
gd_indicator = {gd_indicator}
gd_compare_mode = {gd_compare_mode}
gd_crit = {gd_crit}""")

print(f'=====================\n')

rand_str = f'v{param_ver}_' + datetime.now().strftime("%Y%m%d%H%M%S")
eqn_str = f'{eqn}{{{"-".join(str(c) for c in const)}}}_ftic-{init_data_seed}-{tuned_seed_pair}{"_ic" if allow_ic else ""}{"_anc" if (anchor_budget > 0) else ""}'
train_str = f'nn-{nn_arch}-{hidden_layers}-{hidden_dim}_{optim}_' + ('bcsloss-auto' if autoscale_loss_w_bcs else f'bcsloss-{loss_w_bcs}') + f'_budget-{mem_pts_total_budget}-{num_points}-{anchor_budget}'
method_str = method_str + ('_autoal' if auto_al else '')
folder_name = os.path.join(RESULTS_FOLDER, f'{eqn_str}/{train_str}/{method_str}/' + rand_str)
os.makedirs(folder_name, exist_ok=True)

with open(f'{folder_name}/args', 'w+') as f:
    d_arg = vars(args)
    for k in d_arg.keys():
        f.write(f"{k}: {d_arg[k]}\n")
    
print(f'eqn_str = {eqn_str}')
print(f'train_str = {train_str}')
print('Folder name =', folder_name)

data_code = f'{eqn}-{const[0]}'
model_code = f'{nn_arch}-{hidden_layers}-{hidden_dim}'
param_prefix = f'{data_code}_pb-{init_data_seed}_{model_code}_v{param_ver}'

"""TRAIN INIT MODEL IF NEEDED """

model_path = os.path.join(saved_models_dir, f'{param_prefix}.pkl')

if not os.path.isfile(model_path):
    
    print(f'{model_path} does not exist, need to do training.')
    max_steps = 50000
    
    model, _ = construct_model(
        pde_name=eqn, 
        pde_const=const, 
        use_pdebench=True,
        data_seed=init_data_seed,
        inverse_problem=False,
        data_root=PDEBENCH_DATA,
        num_domain=1000, 
        num_boundary=200, 
        num_initial=200,
        include_ic=allow_ic,
        hidden_layers=hidden_layers, 
        hidden_dim=hidden_dim, 
        activation='tanh', 
        initializer='Glorot uniform', 
        arch=nn_arch, 
    )
    
    train_loop = ModifiedTrainLoop(
        model=model, 
        point_selector_method='random',
        point_selector_args=dict(res_proportion=0.7),
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


""" MODEL SELECT STAGE """

model, model_aux = construct_model(
    
    # problem params
    pde_name=eqn, 
    pde_const=const, 
    use_pdebench=True,  #use_pdebench, 
    data_seed=tuned_seed_pair,
    inverse_problem=False,  #inverse, 
    # inverse_problem_guess=inverse_guess,
    num_domain=num_domain, 
    num_boundary=num_icbc, 
    num_initial=num_icbc,
    include_ic=allow_ic,
    data_root=PDEBENCH_DATA,
    
    # model params
    hidden_layers=hidden_layers, 
    hidden_dim=hidden_dim, 
    activation='tanh', 
    initializer='Glorot uniform', 
    arch=nn_arch, 

)

with open(model_path, 'rb') as f:
    p = pkl.load(f)
    model.params = p
    model.net.params = p[0]


""" DATA SELECT STAGE """

if method == 'random':
    point_selector_method = 'random'
    al_args = dict(
        res_proportion=rand_res_prop,
        method=rand_method,
    )
    
elif method == 'residue':
    point_selector_method = 'residue'
    al_args = dict(
        res_proportion=res_res_prop,
        select_icbc_with_residue=res_all_types,
        select_anc_with_residue=res_all_types,
        k=2.,
        c=0.,
    )
    
elif method in {'greedy', 'kmeans', 'sampling'}:
    point_selector_method = f'eig_{method}'
    al_args = dict(
        weight_method=eig_weight_method, 
        num_points_round=num_points,
        num_candidates_res=800,
        num_candidates_bcs=200,
        num_candidates_init=200,
        sampling=eig_sampling,
        memory=eig_memory,
        scale=eig_scale,
        min_num_points_bcs=1,
        min_num_points_res=1,
        use_init_train_pts=False,
    )
    
# elif method == 'gd':
#     point_selector_method = 'gd'
#     al_args = dict(
#         indicator=gd_indicator,
#         compare_mode=gd_compare_mode,
#         crit=gd_crit,
#         active_eig=10,
#         eig_min=0.1,
#     )
    
else:
    raise ValueError('Invalid method {method}')


""" OPTIMISER SETUP """

if optim == 'adam':
    optim_dict = dict(
        optim_method='adam', 
        optim_lr=5e-5,  # lower for fine-tuning
        train_steps=train_steps,
        snapshot_every=1000,
        al_every=al_every,
        select_anchors_every=al_every,
    )
    
# elif optim == 'lbfgs':
#     optim_dict = dict(
#         optim_method='lbfgs', 
#         optim_lr=1e-2,
#         train_steps=train_steps // 100,
#         snapshot_every=10,
#         al_every=al_every // 100,
#         select_anchors_every=al_every // 100,
#         optim_args=dict(maxiter=100),
#     )
    
else:
    raise ValueError(f'Invalid optim {optim}')


""" TRAIN SETUP """

if point_selector_method.startswith('eig'):
    tensorboard_dir = f'{folder_name}/tensorboard'
    os.makedirs(tensorboard_dir, exist_ok=True)
else:
    tensorboard_dir = None

train_loop = ModifiedTrainLoop(
    model=model, 
    inverse_problem=False,  #inverse,
    point_selector_method=point_selector_method,
    point_selector_args=al_args,
    mem_pts_total_budget=mem_pts_total_budget,
    anchor_budget=anchor_budget,
    loss_w_bcs=loss_w_bcs,
    autoscale_loss_w_bcs=autoscale_loss_w_bcs,
    ntk_ratio_threshold=(0.5 if auto_al else None),
    tensorboard_plots=True,
    log_dir=tensorboard_dir,
    **optim_dict
)


""" TRAIN STAGE """

steps = list(range(0, 100000, 10000)) + list(range(100000, train_steps + 1, 25000))

purge_every = 50000
lb_step = 0
ub_step = purge_every

try:
    
    for i in range(train_steps // purge_every):
        train_loop.train(purge_every)
    
        to_remove = [k for k in train_loop.snapshot_data.keys() if (k is not None) and (k not in steps)]
        for k in to_remove:
            train_loop.snapshot_data.pop(k, None)
            
        reduced_snapshot = dict()
        for k in train_loop.snapshot_data.keys():
            if (k is None) or ((k in steps) and (lb_step <= k <= ub_step)):
                # keep only some snapshots to save memory
                reduced_snapshot[k] = train_loop.snapshot_data[k]
                if 'al_intermediate' in reduced_snapshot[k]:
                    # don't keep jacobians to save memory
                    d_int = reduced_snapshot[k]['al_intermediate']
                    if d_int is not None:
                        d_int.pop('jac_train', None)
                        d_int.pop('jac_candidates', None)

        with open(f'{folder_name}/snapshot_data_s{ub_step}.pkl', 'wb+') as f:
            pkl.dump(reduced_snapshot, f)
            
        with open(f'{folder_name}/al_pts_s{ub_step}.pkl', 'wb+') as f:
            pkl.dump(train_loop.al_data_round, f)
            
        lb_step += purge_every
        ub_step += purge_every

        
except Exception:
    traceback.print_exc()

""" DATA VIS STAGE """

def savefig(fname, fig):
    p = os.path.join(folder_name, fname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    

savefig('losses.png', train_loop.plot_losses()[0])

for i in steps:
    savefig(f'sample/step{i}.png', train_loop.plot_training_data(step_idx=i)[0])
    
for d in range(model.data.test_y.shape[1]):
    
    dim_str = '' if (model.data.test_y.shape[1] == 1) else f'_dim{d}'

    savefig(f'pred/all{dim_str}.png', plot_prediction(train_loop=train_loop, step_idxs=steps, out_idx=d, plot_training_data=True))
    savefig(f'error/all{dim_str}.png', plot_error(train_loop=train_loop, step_idxs=steps, out_idx=d, plot_training_data=True))
    savefig(f'residue/all{dim_str}.png', plot_residue_loss(train_loop=train_loop, step_idxs=steps, plot_training_data=True))

    for i in steps:
        savefig(f'pred/step{i}{dim_str}.png', plot_prediction(train_loop=train_loop, step_idxs=[i], out_idx=d, plot_training_data=False))
        savefig(f'error/step{i}{dim_str}.png', plot_error(train_loop=train_loop, step_idxs=[i], out_idx=d, plot_training_data=False))
        savefig(f'residue/step{i}{dim_str}.png', plot_residue_loss(train_loop=train_loop, step_idxs=[i], plot_training_data=False))
