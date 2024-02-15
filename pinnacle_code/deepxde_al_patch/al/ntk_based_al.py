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


class NTKBasedAL(PointSelector):
    
    def __init__(self, model: dde.Model, points_pool_size: int = 1000, eig_min: float = 1e-2, active_eig: int = None,
                 inverse_problem: bool = False, current_samples: dict = None, 
                 anchor_budget: int = 0, anc_point_filter=None, anc_idx=None,
                 mem_pts_total_budget: int  = None, min_num_points_bcs: int = 0, min_num_points_res: int = 0,
                 loss_w_bcs: float = 1., loss_w_pde: float = 1., loss_w_anc: float = 1., optim_lr: float = 1e-3, enforce_budget: bool =True):
        super().__init__(model=model, inverse_problem=inverse_problem, current_samples=current_samples, 
                         anchor_budget=anchor_budget, anc_point_filter=anc_point_filter, anc_idx=anc_idx,
                         mem_pts_total_budget=mem_pts_total_budget, min_num_points_bcs=min_num_points_bcs, min_num_points_res=min_num_points_res, 
                         loss_w_anc=loss_w_anc, loss_w_bcs=loss_w_bcs, loss_w_pde=loss_w_pde, optim_lr=optim_lr, enforce_budget=enforce_budget)
        self.model = model
        self.data = self.model.data
        self.bcs = self.model.data.bcs
        self.eig_min = eig_min
        self.points_pool_size = points_pool_size
        
        self.points_pool = jnp.array(self.data.geom.random_points(points_pool_size, random='pseudo'))
        self.ntk_fn = NTKHelper(model=model, inverse_problem=inverse_problem)
        # self.jac_all, self.K_fullrank, self.K_reducedrank = self._precompute_pool(eig_min=eig_min) # Not used anymore
        # self.active_eig = active_eig if active_eig else int(jnp.sum(self._use_eig))
        
        self.get_K_subset = None
        self.constrain = None
        self._prepare_functions()
        
    def _precompute_pool(self, eig_min):
        jac_pde = self.ntk_fn.get_jac(jnp.array(self.points_pool), code=-1)
        # jac_bcs = [self.ntk_fn.get_jac(jnp.array(self.points_pool), code=i)
        #            for i in range(len(self.bcs))]
        # jacs = [jac_pde] + jac_bcs
        jac_u = self.ntk_fn.get_jac(jnp.array(self.points_pool), code=-2)
        jacs = [jac_pde, jac_u]
        jac_all = {k: jnp.concatenate([jc[k] for jc in jacs], axis=0) for k in jac_pde.keys()}

        K_fullrank = self.ntk_fn.get_ntk(jac1=jac_all, jac2=jac_all)
        Lambda, Q = jnp.linalg.eigh(K_fullrank)
        use_eig = Lambda > eig_min
        L_sub = Lambda[use_eig]
        Q_sub = Q[:,use_eig]
        K_reducedrank = Q_sub @ jnp.diag(L_sub) @ Q_sub.T
        self._fullrank_Lambda = Lambda
        self._fullrank_Q = Q
        self._use_eig = use_eig
        
        return jac_all, K_fullrank, K_reducedrank
    
    def _prepare_functions(self):
                    
        def _get_K_subset(d):

            jac_pde = self.ntk_fn.get_jac(d['res'], code=-1)
            jac_bcs = [self.ntk_fn.get_jac(d['bcs'][i], code=i) for i in range(len(d['bcs']))]
            
            jacs_sep = [jac_pde] + jac_bcs
            jacs = {k: jnp.concatenate([jc[k] for jc in jacs_sep], axis=0) for k in jac_pde.keys()}
                        
            T_t = self.ntk_fn.get_ntk(jac1=jacs, jac2=jacs)
            eigvals, eigvects = jnp.linalg.eigh(T_t)
            eigvals_sub = eigvals[-self.active_eig:]
            eigvects_sub = eigvects[:,-self.active_eig:]
            T_t_inv = eigvects_sub @ jnp.diag(1. / eigvals_sub) @ eigvects_sub.T
            # T_t_inv = jnp.linalg.inv(T_t + eps * jnp.eye(T_t.shape[0]))
            
            T_nt = self.ntk_fn.get_ntk(jac1=self.jac_all, jac2=jacs)
            K_subset = T_nt @ T_t_inv @ T_nt.T
            
            return K_subset, eigvals_sub, eigvects_sub, jacs
            
        self.get_K_subset = _get_K_subset
            
        constrain_fns = [
            constrain_ic if isinstance(bc, dde.icbc.initial_conditions.IC) else constrain_bc
            for bc in self.data.bcs
        ]
        
        if isinstance(self.data.geom, dde.geometry.GeometryXTime):
            geom = self.data.geom.geometry
            timedomain = self.data.geom.timedomain
        else:
            geom = self.data.geom
            timedomain = None
            
        self._constrain_fns = constrain_fns
        self._geom_obj = geom
        self._timedomain_obj = timedomain
        
        def _constrain(d):
            d['res'] = constrain_domain(points=d['res'], geom=geom, timedomain=timedomain)
            for i in range(len(constrain_fns)):
                d['bcs'][i] = constrain_fns[i](points=d['bcs'][i], geom=geom, timedomain=timedomain)
            return d
        
        self.constrain = _constrain
