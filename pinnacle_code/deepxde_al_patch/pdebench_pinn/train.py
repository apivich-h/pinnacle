""" Modified to work with JAX """
from .. import deepxde as dde
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys
import torch
from typing import Tuple
import h5py

import jax.numpy as jnp

from .utils import (
    PINNDatasetRadialDambreak,
    PINNDatasetDiffReact,
    PINNDataset2D,
    PINNDatasetDiffSorption,
    PINNDatasetBump,
    PINNDataset1Dpde,
    PINNDataset2Dpde,
    PINNDataset3Dpde,
)
from .pde_definitions import (
    pde_diffusion_reaction,
    pde_swe2d,
    # pde_diffusion_sorption,
    # pde_swe1d,
    pde_adv1d,
    pde_diffusion_reaction_1d,
    pde_burgers1D,
    pde_CFD1d,
    pde_CFD2d,
    # pde_CFD3d,
    pde_darcy,
)

# from ..icbc_patch import NeumannBC


# def setup_diffusion_sorption(filename, seed):
#     # TODO: read from dataset config file
#     geom = dde.geometry.Interval(0, 1)
#     timedomain = dde.geometry.TimeDomain(0, 500.0)
#     geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#     D = 5e-4

#     ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
#     bc_d = dde.icbc.DirichletBC(
#         geomtime,
#         lambda x: 1.0,
#         lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
#     )

#     def operator_bc(inputs, outputs, X):
#         # compute u_t
#         du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
#         return outputs - D * du_x

#     bc_d2 = dde.icbc.OperatorBC(
#         geomtime,
#         operator_bc,
#         lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
#     )

#     dataset = PINNDatasetDiffSorption(filename, seed)

#     ratio = int(len(dataset) * 0.3)

#     data_split, _ = torch.utils.data.random_split(
#         dataset,
#         [ratio, len(dataset) - ratio],
#         generator=torch.Generator(device="cpu").manual_seed(42),
#     )

#     data_gt = data_split[:]

#     bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1])

#     data = dde.data.TimePDE(
#         geomtime,
#         pde_diffusion_sorption,
#         [ic, bc_d, bc_d2, bc_data],
#         num_domain=1000,
#         num_boundary=1000,
#         num_initial=5000,
#     )
#     # net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")

#     # def transform_output(x, y):
#     #     return torch.relu(y)

#     # net.apply_output_transform(transform_output)

#     # model = dde.Model(data, net)

#     return data, dataset

def setup_diffusion_reaction(filename, seed, num_domain=1000, num_boundary=1000, num_initial=5000):
    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0, 5.0)
    # timedomain = dde.geometry.TimeDomain(0, 10.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    # bc = NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)


    dataset = PINNDatasetDiffReact(filename, seed)
    initial_input, initial_u, initial_v = dataset.get_initial_condition()

    ic_data_u = dde.icbc.PointSetBC(initial_input, initial_u, component=0)
    ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    # ratio = int(len(dataset) * 0.3)

    # data_split, _ = torch.utils.data.random_split(
    #     dataset,
    #     [ratio, len(dataset) - ratio],
    #     generator=torch.Generator(device="cpu").manual_seed(42),
    # )

    # data_gt = dataset[:]

    # bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    # bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_reaction,
        [bc, ic_data_u, ic_data_v],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )
    data.test_x = jnp.array(dataset.data_input)
    data.test_y = jnp.array(dataset.data_output).reshape(-1, 2)
    # net = dde.nn.FNN([3] + [40] * 6 + [2], "tanh", "Glorot normal")
    # model = dde.Model(data, net)
    
    # data.test_x = data.test_x.at[:,2].set(2. * data.test_x[:,2])

    # return model, dataset
    return data, dataset


def setup_swe_2d(filename, seed) -> Tuple[dde.Model, PINNDataset2D]:

    dataset = PINNDatasetRadialDambreak(filename, seed)

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-2.5, -2.5), (2.5, 2.5))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    # bc = NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic_h = dde.icbc.IC(
        geomtime,
        dataset.get_initial_condition_func(),
        lambda _, on_initial: on_initial,
        component=0,
    )
    ic_u = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=1
    )
    ic_v = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=2
    )

    # ratio = int(len(dataset) * 0.3)

    # data_split, _ = torch.utils.data.random_split(
    #     dataset,
    #     [ratio, len(dataset) - ratio],
    #     generator=torch.Generator(device="cpu").manual_seed(42),
    # )

    # data_gt = data_split[:]
    
    data_gt = dataset[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde_swe2d,
        [bc, ic_h, ic_u, ic_v, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    # net = dde.nn.FNN([3] + [40] * 6 + [3], "tanh", "Glorot normal")
    # model = dde.Model(data, net)
    
    nan_list = np.nan * np.ones(shape=(dataset.data_output.shape[0], 2))
    test_y = np.concatenate((dataset.data_output.reshape(-1, 1), nan_list), axis=1)

    data.test_x = jnp.array(dataset.data_input)
    data.test_y = jnp.array(test_y)
    
    return data, dataset

def _boundary_r(x, on_boundary, xL, xR):
    return (on_boundary and jnp.isclose(x[0], xL)) or (on_boundary and jnp.isclose(x[0], xR))

def setup_pde1D(filename="1D_Advection_Sols_beta0.1.hdf5",
                root_path='data',
                val_batch_idx=0,
                input_ch=2,
                output_ch=1,
                hidden_ch=40,
                xL=0.,
                xR=1.,
                aux_params=None,
                inverse_problem=False,
                inverse_problem_guess=None,
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000):

    # TODO: read from dataset config file
    geom = dde.geometry.Interval(xL, xR)
    boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, xL, xR)
    
    if 'ReacDiff' in filename:
        # timedomain = dde.geometry.TimeDomain(0, 2.0)
        timedomain = dde.geometry.TimeDomain(0, 1.0)
        _pde = pde_diffusion_reaction_1d
        if_periodic_bc = True
        
    elif 'Advection' in filename:
        # timedomain = dde.geometry.TimeDomain(0, 2.0)
        timedomain = dde.geometry.TimeDomain(0, 4.0)
        _pde = pde_adv1d
        if_periodic_bc = True
        
    elif 'Burgers' in filename:
        # timedomain = dde.geometry.TimeDomain(0, 2.0)
        timedomain = dde.geometry.TimeDomain(0, 4.0)
        _pde = pde_burgers1D
        if_periodic_bc = True
        
    elif 'CFD' in filename:
        # timedomain = dde.geometry.TimeDomain(0, 1.0)
        timedomain = dde.geometry.TimeDomain(0, 2.0)
        _pde = pde_CFD1d
        if_periodic_bc = ('periodic' in filename)
    
    else:
        raise ValueError('Filename invalid')
    
    if inverse_problem:
        assert len(inverse_problem_guess) == len(aux_params)
        ext_param = [dde.Variable(v) for v in inverse_problem_guess]
        pde = lambda x, y, const: _pde(x, y, *const)
    else:
        ext_param = None
        pde = lambda x, y: _pde(x, y, *aux_params)
    
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset1Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    
    if 'CFD' in filename:
        ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,0].unsqueeze(1), component=0)
        ic_data_v = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,1].unsqueeze(1), component=1)
        ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,2].unsqueeze(1), component=2)
    else:
        ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u, component=0)
        
    # prepare boundary condition
    if if_periodic_bc:
        if 'CFD' in filename:
            bc_D = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            bc_V = dde.icbc.PeriodicBC(geomtime, 1, boundary_r)
            bc_P = dde.icbc.PeriodicBC(geomtime, 2, boundary_r)

            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_d, ic_data_v, ic_data_p, bc_D, bc_V, bc_P],
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
            )
        else:
            bc = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_u, bc],
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
            )
    else:
        ic = dde.icbc.IC(
            geomtime, lambda x: -jnp.sin(jnp.pi * x[:, 0:1]), lambda _, on_initial: on_initial
        )
        bd_input, bd_uL, bd_uR = dataset.get_boundary_condition()
        bc_data_uL = dde.icbc.PointSetBC(bd_input.cpu(), bd_uL, component=0)
        bc_data_uR = dde.icbc.PointSetBC(bd_input.cpu(), bd_uR, component=0)

        data = dde.data.TimePDE(
            geomtime,
            pde,
            [ic, bc_data_uL, bc_data_uR],
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=num_initial,
        )
        
    data.test_x = jnp.array(dataset.data_input)
    data.test_y = jnp.array(dataset.data_output)
    
    # correct a mistake in some 1d examples
    # if ('Advection' in filename):  # or ('Burgers' in filename):
    if True:
        data.test_x = data.test_x.at[:,1].set(2. * data.test_x[:,1])
    
    # net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    # model = dde.Model(data, net)
    
    # print(dataset.data_input, dataset.data_output)

    return data, ext_param, dataset

def setup_darcy(filename="2D_DarcyFlow_beta0.01_Train.hdf5",
                root_path='data',
                val_batch_idx=-1,
                inverse_problem=False,
                inverse_problem_guess=None,
                aux_params=[0.01],
                num_domain=5000,
                num_boundary=1000,
                num_initial=1000,):
    
    f1 = h5py.File(filename, "r")

    a = np.array(f1['nu'][val_batch_idx])
    u = np.array(f1['tensor'][val_batch_idx, 0])
    zs = np.stack([a.flatten(), u.flatten()]).T
    
    x = jnp.array(f1['x-coordinate'])
    y = jnp.array(f1['y-coordinate'])
    grid = jnp.meshgrid(x, y)
    xy = jnp.array(grid).reshape(2, -1).T

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((0, 0), (1, 1))
    # timedomain = dde.geometry.TimeDomain(0., 1.0)
    # pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    # geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    if inverse_problem:
        assert len(inverse_problem_guess) == len(aux_params)
        ext_param = [dde.Variable(v) for v in inverse_problem_guess]
        pde = lambda x, y, const: pde_darcy(x, y, *const)
    else:
        ext_param = None
        pde = lambda x, y: pde_darcy(x, y, *aux_params)

    # dataset = PINNDataset2Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # # prepare initial condition
    # initial_input, initial_u = dataset.get_initial_condition()
    # ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    # ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    # ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    # ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    # # prepare boundary condition
    
    bc = dde.icbc.DirichletBC(geom, lambda xs: 0., lambda _, on_boundary: on_boundary, component=1)
    
    xy = jnp.array(xy)
    zs = jnp.array(zs)
    ic_sample_idx = jnp.array(np.random.default_rng(seed=42).choice(xy.shape[0], size=5000, replace=False))
    obs_x = xy[ic_sample_idx, :]
    obs_y = zs[ic_sample_idx, 0:1]
    ic = dde.icbc.PointSetBC(obs_x, obs_y, component=0)
    
    data = dde.data.PDE(
        geom,
        pde,
        [bc, ic],
        num_domain=num_domain,
        num_boundary=num_boundary
    )
    data.test_x = xy
    data.test_y = zs
    
    # net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    # model = dde.Model(data, net)

    return data, ext_param, None

# def setup_CFD2D(filename="2D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
#                 root_path='data',
#                 val_batch_idx=-1,
#                 input_ch=2,
#                 output_ch=4,
#                 hidden_ch=40,
#                 xL=0.,
#                 xR=1.,
#                 yL=0.,
#                 yR=1.,
#                 if_periodic_bc=True,
#                 aux_params=[1.6667],
#                 num_domain=5000,
#                 num_boundary=1000,
#                 num_initial=1000,):

#     # TODO: read from dataset config file
#     geom = dde.geometry.Rectangle((-1, -1), (1, 1))
#     timedomain = dde.geometry.TimeDomain(0., 1.0)
#     pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
#     geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#     dataset = PINNDataset2Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
#     # prepare initial condition
#     initial_input, initial_u = dataset.get_initial_condition()
#     ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
#     ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
#     ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
#     ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
#     # prepare boundary condition
#     bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
#     data = dde.data.TimePDE(
#         geomtime,
#         pde,
#         [ic_data_d, ic_data_vx, ic_data_vy, ic_data_p],#, bc],
#         num_domain=num_domain,
#         num_boundary=num_boundary,
#         num_initial=num_initial,
#     )
#     # net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
#     # model = dde.Model(data, net)

#     return data, dataset

# def setup_CFD3D(filename="3D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
#                 root_path='data',
#                 val_batch_idx=-1,
#                 input_ch=2,
#                 output_ch=4,
#                 hidden_ch=40,
#                 aux_params=[1.6667]):

#     # TODO: read from dataset config file
#     geom = dde.geometry.Cuboid((0., 0., 0.), (1., 1., 1.))
#     timedomain = dde.geometry.TimeDomain(0., 1.0)
#     pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
#     geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#     dataset = PINNDataset3Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
#     # prepare initial condition
#     initial_input, initial_u = dataset.get_initial_condition()
#     ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
#     ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
#     ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
#     ic_data_vz = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
#     ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,4].unsqueeze(1), component=4)
#     # prepare boundary condition
#     bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
#     data = dde.data.TimePDE(
#         geomtime,
#         pde,
#         [ic_data_d, ic_data_vx, ic_data_vy, ic_data_vz, ic_data_p, bc],
#         num_domain=1000,
#         num_boundary=1000,
#         num_initial=5000,
#     )
#     net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
#     model = dde.Model(data, net)

#     return model, dataset
