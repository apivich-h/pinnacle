import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import tqdm

import jax.numpy as jnp

from . import deepxde as dde

from .modified_train_loop import ModifiedTrainLoop
from .ntk import NTKHelper


def _plot_grid_timesteps(step_idxs, xs, zs_arrs, train_loop, res=100, plot_training_data=True):
    xi, yi = [np.linspace(np.min(xs[:,i]), np.max(xs[:,i]), res) for i in range(2)]
    grid = np.meshgrid(xi, yi)
    fig, axs = plt.subplots(nrows=1, ncols=len(step_idxs), figsize=(plt.rcParams['figure.figsize'][0] * (len(step_idxs) + 0.25), plt.rcParams['figure.figsize'][0]))
    if len(step_idxs) == 1:
        axs = [axs]
    levels = np.linspace(np.min(zs_arrs), np.max(zs_arrs), num=2*res)
    
    if plot_training_data:
        for ax, step_idx in zip(axs, step_idxs):
            train_loop.plot_training_data(step_idx=step_idx, ax=ax)
        
    triang = tri.Triangulation(xs[:,0], xs[:,1])
    for ax, step_idx, z in zip(axs, step_idxs, zs_arrs):
        interpolator = tri.LinearTriInterpolator(triang, z.flatten())
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        cb = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
        ax.set_title(f'Step {step_idx}')
        
    fig.colorbar(cb, ax=list(axs))
    return fig


def plot_residue_loss(train_loop, step_idxs, res=100, plot_training_data=True, t_plot=0.):
    in_dim = train_loop.x_test.shape[1]
    if in_dim <= 2:
        xs = train_loop.x_test
    elif in_dim == 3:
        assert t_plot is not None
        xs = train_loop.x_test[train_loop.x_test[:,2] == t_plot]
        assert xs.shape[0] > 0
    else:
        raise ValueError('in_dim > 3')
    zs_arrs = [np.abs(train_loop.pde_residue(xs=xs, step_idx=s)) for s in step_idxs]
    return _plot_grid_timesteps(step_idxs=step_idxs, xs=xs, zs_arrs=zs_arrs, train_loop=train_loop, res=res, plot_training_data=plot_training_data)


def plot_error(train_loop, step_idxs, res=100, out_idx=0, plot_training_data=True, t_plot=None):
    in_dim = train_loop.x_test.shape[1]
    if in_dim <= 2:
        xs = train_loop.x_test
        ys = train_loop.y_test
    elif in_dim == 3:
        assert t_plot is not None
        idxs = train_loop.x_test[:,2] == t_plot
        xs = train_loop.x_test[idxs, :]
        ys = train_loop.y_test[idxs, out_idx]
        assert xs.shape[0] > 0
    else:
        raise ValueError('in_dim > 3')
    zs_arrs = [np.abs(ys - train_loop.solution_prediction(xs, step_idx=s)[:,out_idx].reshape(*ys.shape)) for s in step_idxs]
    return _plot_grid_timesteps(step_idxs=step_idxs, xs=xs, zs_arrs=zs_arrs, train_loop=train_loop, res=res, plot_training_data=plot_training_data)


def plot_prediction(train_loop, step_idxs=None, res=100, out_idx=0, plot_training_data=True, t_plot=None):
    has_step_idxs = ((step_idxs is not None) and (len(step_idxs) > 0))
    if not has_step_idxs:
        step_idxs = []
    
    in_dim = train_loop.x_test.shape[1]
    if in_dim <= 2:
        xs = train_loop.x_test
        z_actual = train_loop.y_test[:, out_idx].flatten()
        if has_step_idxs:
            zs_arrs = [train_loop.solution_prediction(xs, step_idx=s)[:,out_idx].flatten() for s in step_idxs]
        else:
            zs_arrs = []
    elif in_dim == 3:
        assert t_plot is not None
        idxs = (train_loop.x_test[:,2] == t_plot)
        xs = train_loop.x_test[idxs, :]
        assert xs.shape[0] > 0
        z_actual = train_loop.y_test[idxs, out_idx].flatten()
        if has_step_idxs:
            zs_arrs = [train_loop.solution_prediction(xs, step_idx=s)[:,out_idx].flatten() for s in step_idxs]
        else:
            zs_arrs = []
    else:
        raise ValueError('in_dim > 3')
    
    xi, yi = [np.linspace(np.min(xs[:,i]), np.max(xs[:,i]), res) for i in range(2)]
    grid = np.meshgrid(xi, yi)
    fig, axs = plt.subplots(nrows=1, ncols=len(step_idxs) + 1, figsize=(plt.rcParams['figure.figsize'][0] * (len(step_idxs) + 1.25), plt.rcParams['figure.figsize'][0]))
    if not has_step_idxs:
        axs = [axs]
    
    triang = tri.Triangulation(xs[:,0], xs[:,1])
    
    levels = np.linspace(np.min([z_actual] + zs_arrs), np.max([z_actual] + zs_arrs), num=2*res)
    if has_step_idxs:
        
        if plot_training_data and (train_loop.x_test.shape[1] == 2):
            for ax, step_idx in zip(axs[1:], step_idxs):
                train_loop.plot_training_data(step_idx=step_idx, ax=ax)
        
        for ax, step_idx, z in zip(axs[1:], step_idxs, zs_arrs):
            interpolator = tri.LinearTriInterpolator(triang, z.flatten())
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            cb = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
            ax.set_title(f'Step {step_idx}')
        
    interpolator = tri.LinearTriInterpolator(triang, z_actual)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    cb = axs[0].contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
    axs[0].set_title(f'True solution')
        
    if has_step_idxs:
        axs = axs.ravel().tolist()
    fig.colorbar(cb, ax=axs)
    return fig

# Plot eigenvalues
def plot_eigvals(train_loop,step_idx):
    fig = plt.plot(jnp.log10(train_loop.snapshot_data[step_idx]['al_intermediate']['eigvals']))
    plt.title(f'Eigenvalues of NTK at step {step_idx}')
    plt.xlabel('Index')
    plt.ylabel('Log10 of eigenvalues')
    return fig

# Eigenbasis plots differentiating between residual and predict

def func_kernel_pred(train_loop,x_test,step_idx,idx=-1,code = -2, use_const_res=True):
    ntk_fn = NTKHelper(train_loop.model)
    jacs_x_test = ntk_fn.get_jac(x_test, code=code)
    jacs_train = train_loop.snapshot_data[step_idx]['al_intermediate']['jac_train']
    K_train_x_test = ntk_fn.get_ntk(jac1=jacs_train, jac2=jacs_x_test)

    # K_train_x_test = train_loop.snapshot_data[step_idx]['al_intermediate']['K_train_test']
    eigvects = train_loop.snapshot_data[step_idx]['al_intermediate']['eigvects']
    eigvals = train_loop.snapshot_data[step_idx]['al_intermediate']['eigvals']
    residual_old = train_loop.snapshot_data[step_idx]['al_intermediate']['residual_old']
    
    # # Selecting specific eigenvectors
    if use_const_res == True:
        # output1 = jnp.ones(residual_old.shape) @eigvects[:,idx]
        # Correct approach is to not use use residuals. 
        output1 = 1
    else:
        output1 = residual_old@eigvects[:,idx]
    output = output1 * 1/eigvals[idx]* eigvects[:,idx].T @ K_train_x_test
    return output

def plot_eigenbasis(train_loop,step_idx=0, num_plots=10, plots_per_level = 5, use_const_res=True):
    # Plotting eigenbasis

    # Plotting grid settings
    res = 30
    xs = train_loop.x_test
    xi, yi = [np.linspace(np.min(xs[:,i]), np.max(xs[:,i]), res) for i in range(2)]
    grid = np.meshgrid(xi, yi)
    pool_pts = jnp.array(grid).reshape(2, -1).T
    x_test = pool_pts

    fig, axs = plt.subplots(2, plots_per_level, figsize=(30, 15))
    for i in range(plots_per_level):
        idx = -1-i
        # First level is for predict, second is for residual
        x_train = train_loop.snapshot_data[step_idx]['samples']

        samples = x_train
        for level in range(2):
            axs[level,i%plots_per_level].plot(samples['res'][:, 0], samples['res'][:, 1], 'o')
            for bc_pts in samples['bcs']:
                axs[level,i%plots_per_level].plot(bc_pts[:, 0], bc_pts[:, 1], '^')
            if level == 0:
                axs[level,i% plots_per_level].set_title(f'Top {-idx} Eigenvector for predict, step {step_idx}, Eigval = {train_loop.snapshot_data[step_idx]["al_intermediate"]["eigvals"][idx]:.2f}')
                T = func_kernel_pred(train_loop,x_test = x_test,step_idx=step_idx, idx=-idx, use_const_res=use_const_res, code = -2).reshape(res, res)
            else:
                axs[level,i% plots_per_level].set_title(f'Top {-idx} Eigenvector for residual, step {step_idx}, Eigval = {train_loop.snapshot_data[step_idx]["al_intermediate"]["eigvals"][idx]:.2f}')
                T = func_kernel_pred(train_loop,x_test = x_test,step_idx=step_idx, idx=-idx, use_const_res=use_const_res, code = -1).reshape(res, res)

            cb = axs[level,i%plots_per_level].pcolormesh(*grid, T, cmap='RdBu_r')

            fig.colorbar(cb, ax=axs[level,i%plots_per_level])
    return fig


def plot_inv(train_loop):
    inv_est = plt.plot([train_loop.snapshot_data[i]['params'][1] for i in train_loop.snapshot_data.keys()], label = 'estimated value')
    # plotting constant line of true value pde_constant
    plt.plot([pde_const[0] for i in train_loop.snapshot_data.keys()], label = 'true value')
    plt.title("Estimate of PDE parameter")
    plt.legend()
    # plt.savefig(save_path + fn + "_inv_est.png")
    plt.close()
    return inv_est


def plot_all_eigvals(train_loop):
    # Eigenvalue plots
    step_idx_max = max(train_loop.snapshot_data.keys())
    num_plots = step_idx_max // train_loop.al_every + 1 + 1# last +1 is to add final step

    for i in range(num_plots):
            if i == (num_plots-1):
                step_idx = step_idx_max # final step
            else: 
                step_idx = train_loop.al_every * i 

            
            # plt.plot(jnp.log10(train_loop.snapshot_data[step_idx]['al_intermediate']['eigvals']), label=f'step_idx={step_idx}')
            plt.semilogy(train_loop.snapshot_data[step_idx]['al_intermediate']['eigvals'], label=f'step_idx={step_idx}')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'Eigenvalues of NTK at various steps')
    plt.xlabel('Index')
    plt.ylabel('Log10 of eigenvalues')        

# import matplotlib.pyplot as plt
# import numpy as np
# import tqdm

# import jax.numpy as jnp

# import deepxde as dde

# from .modified_train_loop import ModifiedTrainLoop


# def _plot_grid_timesteps(step_idxs, grid, zs, train_loop):
#     shape = grid[0].shape
#     fig, axs = plt.subplots(nrows=1, ncols=len(step_idxs), figsize=(8 * len(step_idxs) + 4, 8))
#     zmin, zmax = np.min(zs), np.max(zs)
#     for ax, step_idx in zip(axs, [0] + step_idxs[:-1]):
#         train_loop.plot_training_data(step_idx=step_idx, ax=ax)
#     for ax, step_idx, z in zip(axs, step_idxs, zs):
#         cb = ax.pcolormesh(*grid, z, cmap='RdBu_r', vmin=zmin, vmax=zmax)
#         ax.set_title(f'Step {step_idx}')
#     fig.colorbar(cb, ax=axs.ravel().tolist())
#     return fig


# def plot_residue_loss(train_loop, step_idxs, grid):
#     pool_pts = jnp.array(grid).reshape(2, -1).T
#     shape = grid[0].shape
#     zs = [np.abs(train_loop.pde_residue(xs=pool_pts, step_idx=s)).reshape(*shape) for s in step_idxs]
#     return _plot_grid_timesteps(step_idxs=step_idxs, grid=grid, zs=zs, train_loop=train_loop)


# def plot_error(train_loop, step_idxs, grid):
#     pool_pts = jnp.array(grid).reshape(2, -1).T
#     shape = grid[0].shape
#     true_soln = train_loop.data.soln(pool_pts).reshape(*shape)
#     zs = [np.abs(true_soln - train_loop.solution_prediction(pool_pts, step_idx=s)[:,0].reshape(*shape)) for s in step_idxs]
#     return _plot_grid_timesteps(step_idxs=step_idxs, grid=grid, zs=zs, train_loop=train_loop)


# def plot_prediction(train_loop, step_idxs, grid):
#     pool_pts = jnp.array(grid).reshape(2, -1).T
#     shape = grid[0].shape
#     true_soln = train_loop.data.soln(pool_pts).reshape(*shape)
#     zs = [train_loop.solution_prediction(pool_pts, step_idx=s)[:,0].reshape(*shape) for s in step_idxs]
#     shape = grid[0].shape
#     fig, axs = plt.subplots(nrows=1, ncols=len(step_idxs) + 1, figsize=(8 * len(step_idxs) + 12, 8))
#     zmin, zmax = np.min(true_soln), np.max(true_soln)
#     for ax, step_idx in zip(axs[1:], [0] + step_idxs[:-1]):
#         train_loop.plot_training_data(step_idx=step_idx, ax=ax)
#     for ax, step_idx, z in zip(axs[1:], step_idxs, zs):
#         cb = ax.pcolormesh(*grid, z, cmap='RdBu_r', vmin=zmin, vmax=zmax)
#         ax.set_title(f'Step {step_idx}')
#     cb = axs[0].pcolormesh(*grid, true_soln, cmap='RdBu_r', vmin=zmin, vmax=zmax)
#     axs[0].set_title(f'True solution')
#     fig.colorbar(cb, ax=axs.ravel().tolist())
#     return fig
