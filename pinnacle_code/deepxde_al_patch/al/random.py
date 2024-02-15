from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

from .. import deepxde as dde

from ..ntk import NTKHelper
from ..icbc_patch import constrain_domain, constrain_ic, constrain_bc
from .al_pinn import PointSelector
from ..utils import pairwise_dist


class RandomPointSelector(PointSelector):
    
    def __init__(self, model: dde.Model, res_proportion: float = 0.5, method='pseudo',
                 inverse_problem: bool = False, current_samples: dict = None, 
                 anchor_budget: int = 0, anc_point_filter=None, anc_idx=None,
                 mem_pts_total_budget: int  = None, min_num_points_bcs: int = 0, min_num_points_res: int = 0,
                 loss_w_bcs: float = 1., loss_w_pde: float = 1., loss_w_anc: float = 1., optim_lr: float = 1e-3):
        super().__init__(model=model, inverse_problem=inverse_problem, current_samples=current_samples, 
                         anchor_budget=anchor_budget, anc_point_filter=anc_point_filter, anc_idx=anc_idx,
                         mem_pts_total_budget=mem_pts_total_budget, min_num_points_bcs=min_num_points_bcs, min_num_points_res=min_num_points_res, 
                         loss_w_anc=loss_w_anc, loss_w_bcs=loss_w_bcs, loss_w_pde=loss_w_pde, optim_lr=optim_lr)
        self.model = model
        self.data = self.model.data
        self.bcs = self.model.data.bcs
        self.res_proportion = res_proportion
        self.method = method
        
    def generate_samples(self):
        
        n_anc_total = self.anchor_budget + (
            self.current_samples['anc'].shape[0] if self.current_samples is not None and ('anc' in self.current_samples)
            else 0
        )
        n_res = int(self.res_proportion * self.mem_pts_total_budget) if (len(self.bcs) > 0) else (self.mem_pts_total_budget)
        n_per_bc = (self.mem_pts_total_budget - n_res) // len(self.bcs) if (len(self.bcs) > 0) else 0
        
        print(f'Will select {n_res} collocation points.')
        print(f'Will select {n_per_bc} points for each of the {len(self.bcs)} boundary conditions.')
        print(f'Will select {self.anchor_budget} extra anchors, giving total of {n_anc_total} anchors.')
        
        returned_pts = {
            'res': jnp.array(self.data.geom.random_points(n_res, random=self.method)),
            'bcs': []
        }
        for bc in self.data.bcs:
            if isinstance(bc, dde.icbc.boundary_conditions.PointSetBC):
                idxs = jnp.array(np.random.choice(a=bc.points.shape[0], size=n_per_bc, replace=False))
                xs = bc.points[idxs, :]
            elif isinstance(bc, dde.icbc.initial_conditions.IC):
                xs = jnp.array(self.data.geom.random_initial_points(n_per_bc))
            else:
                xs = jnp.array(self.data.geom.random_boundary_points(n_per_bc))
            returned_pts['bcs'].append(xs)
        
        if self.anchor_budget > 0:
            anc_pts = jnp.array(self.data.geom.random_points(self.anchor_budget, random='pseudo'))
            anc_candidate = self.anc_point_filter(jnp.array(self.data.train_x_all))
            dist = pairwise_dist(anc_pts, anc_candidate)
            closest_pts = jnp.argmin(dist, axis=1)
            assert anc_pts.shape[0] == closest_pts.shape[0], closest_pts
            returned_pts['anc'] = jnp.array([anc_candidate[i] for i in closest_pts])
        
        if self.current_samples is not None and ('anc' in self.current_samples):
            if self.anchor_budget > 0:
                returned_pts['anc'] = jnp.concatenate([self.current_samples['anc'], returned_pts['anc']], axis=0)
            else:
                returned_pts['anc'] = self.current_samples['anc']
        
        return returned_pts, {'chosen_pts': returned_pts}