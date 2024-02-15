from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
import random

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
from ..icbc_patch import (constrain_bc, constrain_domain, constrain_ic,
                          generate_residue)
from .al_pinn import PointSelector
from ..utils import pairwise_dist


class ResidueSelector(PointSelector):
    
    def __init__(self, model: dde.Model, res_proportion: float = 0.5, k: float = 2., c: float = 0.,
                 select_icbc_with_residue: bool = True, select_anc_with_residue: bool = True,
                 inverse_problem: bool = False, current_samples: dict = None, 
                 anchor_budget: int = 0, unlimited_colloc_pts: bool = False, anc_point_filter=None, anc_idx=None, target_fn_param=None,
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
        self.unlimited_colloc_pts = unlimited_colloc_pts
        self.k = k
        self.c = c
        self.select_icbc_with_residue = select_icbc_with_residue
        self.select_anc_with_residue = select_anc_with_residue
        self.target_fn_param = target_fn_param
        
    def generate_samples(self):
        
        aux_dict = dict()
        
        def _pde_residue_fn(params, xs):
            if self.inverse_problem:
                nn_params, ext_params = params
                f_ = lambda xs: self.model.net.apply(nn_params, xs, training=False)
                return self.data.pde(xs, (f_(xs), f_), ext_params)[0].reshape(-1)
            else:
                f_ = lambda xs: self.model.net.apply(params[0], xs, training=False)
                return self.data.pde(xs, (f_(xs), f_))[0].reshape(-1)
            
        def gen_loss_fn(params, xs, bc):
            f = generate_residue(bc, net_apply=self.model.net.apply)
            return f(params[0], xs)
        
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
            'res': None,
            'bcs': []
        }
        
        res_pool = jnp.array(self.data.geom.random_points(5 * n_res, random='pseudo'))
        residual = _pde_residue_fn(self.model.params, res_pool)
        aux_dict['res_pool'] = res_pool
        aux_dict['residual'] = residual
        
        probs = np.array(residual ** self.k) + self.c + 1e-9
        n = probs.shape[0]
        point_score = np.empty(shape=(n,))
        points = []
        for i in range(n_res):
            p = random.choices(population=list(range(n)), weights=probs)[0]
            probs[p] = 0.
            points.append(res_pool[p])
        returned_pts['res'] = jnp.array(points)
        
        if self.select_icbc_with_residue:
            
            icbc_details = []
            
            for bc in self.data.bcs:
                if isinstance(bc, dde.icbc.boundary_conditions.PointSetBC):
                    xs_pool = bc.points
                elif isinstance(bc, dde.icbc.initial_conditions.IC):
                    xs_pool = jnp.array(self.data.geom.random_initial_points(5*n_per_bc))
                else:
                    xs_pool = jnp.array(self.data.geom.random_boundary_points(5*n_per_bc))
                    
                error = _pde_residue_fn(self.model.params, xs_pool)
                if len(error.shape) > 1:
                    error = jnp.sum(error, axis=1)
                icbc_details.append((xs_pool, error))
                
                probs = np.array(error ** self.k) + self.c + 1e-9
                n = probs.shape[0]
                point_score = np.empty(shape=(n,))
                points = []
                for i in range(n_per_bc):
                    p = random.choices(population=list(range(n)), weights=probs)[0]
                    probs[p] = 0.
                    points.append(xs_pool[p])
                    
                returned_pts['bcs'].append(jnp.array(points))
        
        else:
            
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
            
            if self.select_anc_with_residue:
                anc_pool = self.anc_point_filter(jnp.array(self.data.train_x_all))
                if self.target_fn_param is not None:
                    out_anc = self.model.net.apply(self.model.params[0], anc_pool, training=False)
                    pseudo_target = self.model.net.apply(self.target_fn_param[0], anc_pool, training=False)
                    anc_residual = out_anc - pseudo_target
                    aux_dict['anc_pseudo'] = pseudo_target
                    aux_dict['anc_out'] = out_anc
                else:
                    anc_residual = _pde_residue_fn(self.model.params, anc_pool)
                aux_dict['anc_pool'] = anc_pool
                aux_dict['anc_residual'] = anc_residual
            
                probs = np.array(anc_residual ** self.k) + self.c + 1e-9
                probs = probs[:,0]
                n = probs.shape[0]
                point_score = np.empty(shape=(n,))
                anc_points = []
                for i in range(self.anchor_budget):
                    p = random.choices(population=list(range(n)), weights=probs)[0]
                    probs[p] = 0.
                    anc_points.append(anc_pool[p])
                
                returned_pts['anc'] = jnp.array(anc_points)
                # returned_pts['anc'] = jnp.array(self.data.geom.random_points(self.anchor_budget, random=self.method))
                
            else:
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
                
        # in case we can just grow collocation points however
        if self.unlimited_colloc_pts and (self.current_samples is not None):
            returned_pts['res'] = jnp.concatenate([self.current_samples['res'], returned_pts['res']], axis=0)
            for i in range(len(returned_pts['bcs'])):
                returned_pts['bcs'][i] = jnp.concatenate([self.current_samples['bcs'][i], returned_pts['bcs'][i]], axis=0)
                
        aux_dict['chosen_pts'] = returned_pts
        return returned_pts, aux_dict