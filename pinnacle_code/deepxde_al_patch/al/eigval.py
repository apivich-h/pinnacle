import os
import pickle as pkl
from collections.abc import MutableMapping
from functools import partial
import random

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm
from flax import linen as nn
# import cvxpy as cp
from sklearn.cluster._kmeans import kmeans_plusplus

from .. import deepxde as dde

from ..icbc_patch import (constrain_bc, constrain_domain, constrain_ic,
                          generate_residue)
from ..ntk import NTKHelper
from ..utils import dict_pts_size, flatten_pts_dict, to_cpu
from .ntk_based_al import NTKBasedAL


class EigenvaluePointSelector(NTKBasedAL):
    # TODO add in parameters to select different BCs and residual points
    def __init__(self, model: dde.Model, 
                 inverse_problem: bool = False,
                 current_samples: dict = None,
                 selection_method: str = 'greedy',
                 weight_method: str = "labels", # possible options are 'none', 'labels', 'eigvals', 'labels_train'
                 mem_pts_total_budget: int = None, # total number of points to select
                 num_points_round: int = 30,
                 min_num_points_bcs: int = 0,
                 min_num_points_res: int = 0,
                 num_candidates_res: int=60,
                 num_candidates_bcs: int=10,
                 num_candidates_init: int=10,
                 anc_point_filter=None,
                 use_anc_in_train=True,
                 anc_idx=None,
                 dupl_pts_res: bool = True, # add anchor points and BCs candidates into res candidates
                 memory: bool = True, # remember old samples and add new ones to them
                 use_random_base = True, # use random base when computing the reference NTK
                #  mem_pts_ratio: float = 1, # num points to remember from past iterations. If enforce_budget is True, then this will be dyanmically selected when budget is hit.
                 include_anc_cap: bool = False, # include anchor points in the pool
                 mem_refine_selection_method: str = 'random', # method to select points to remember
                 mem_refine_weight_method: str = 'random',
                 use_init_train_pts: bool = True, # use initial training points in the pool
                 sampling: str = 'pseudo',
                 loss_w_bcs: float = [1.0,1.0],  
                 loss_w_pde: float = 1.0, 
                 loss_w_anc: float = 1.0,
                 anchor_budget: int = 0,
                 scale: str = 'none', # Options are 'none', 'max', 'trace'
                 normalise_N: bool = False, # Normalise with number of points. Not fully implemented for 'trace' scaling
                 lr_cap: bool = False,
                 enforce_budget: bool = True,
                 optim_lr : float = 1e-3,
                 points_pool_size: int = 1000, 
                 active_eig: int = None,
                 eig_min: float = 1e-4,
                 eps_ntk: float = 1e-8,
                 target_fn_param=None):
        super().__init__(
            model=model, points_pool_size=points_pool_size, eig_min=eig_min, active_eig=active_eig,
            inverse_problem=inverse_problem, current_samples=current_samples, 
            anchor_budget=anchor_budget, anc_point_filter=anc_point_filter, anc_idx=anc_idx,
            mem_pts_total_budget=mem_pts_total_budget, min_num_points_bcs=min_num_points_bcs, min_num_points_res=min_num_points_res, 
            loss_w_anc=loss_w_anc, loss_w_bcs=loss_w_bcs, loss_w_pde=loss_w_pde, optim_lr=optim_lr, enforce_budget=enforce_budget,
        )
        self.selection_method = selection_method
        self.weight_method = weight_method # possible options are 'none', 'labels', 'eigvals'
        self.use_random_base = use_random_base
        self.memory = memory
        # self.mem_pts_ratio = mem_pts_ratio
        self.include_anc_cap = include_anc_cap
        self.mem_refine_selection_method = mem_refine_selection_method
        self.mem_refine_weight_method = mem_refine_weight_method
        self.use_init_train_pts = use_init_train_pts
        self.num_points_round = num_points_round
        self.num_candidates_res = num_candidates_res
        self.num_candidates_bcs = num_candidates_bcs
        self.num_candidates_init = num_candidates_init
        self.dupl_pts_res = dupl_pts_res
        self.sampling = sampling
        self.scale = scale
        self.normalise_N = normalise_N
        self.lr_cap = lr_cap
        self.eps_ntk = eps_ntk
        self.use_anc_in_train = use_anc_in_train
        self.target_fn_param = target_fn_param

    # Helper function to filter dictionary of datapoints corresponding to indices of flattened dictionary
    # Need at least Python 3.7 for this to work, as it assumes that the order of keys in a dictionary is preserved
    def filter_dict(self,dict,idx):
        counter = 0
        new_dict = {}
        for key, value in dict.items():
            if isinstance(value,(jnp.ndarray)) or isinstance(value,(np.ndarray)):
                new_array_list = []
                for i in range(value.shape[0]):
                    if counter in idx:
                        new_array_list.append(value[i])
                    counter += 1
                new_array = jnp.array(new_array_list)
                new_dict[key] = new_array
                new_array = []
                
            if isinstance(value,list):
                new_list = []
                for i in range(len(value)):
                    if isinstance(value[i],(jnp.ndarray)) or isinstance(value[i],(np.ndarray)):
                        new_array_list = []
                        for j in range(value[i].shape[0]):
                            if counter in idx:
                                new_array_list.append(value[i][j])
                            counter += 1
                        new_array = jnp.array(new_array_list)
                    if len(new_array) != 0:
                        new_list.append(new_array)  
                new_dict[key] = new_list
        return new_dict
    
    def generate_samples(self):
        
        # Now directly computing required NTKs. TODO make use of superclass methods

        # Initial training data that is used as a base to compute kernel. 
        
        # if True:
        if (not self.memory) or (self.current_samples is None) or self.use_random_base:
            d = {
                'res': jnp.array(self.data.train_x_all),
                'bcs': [],
            }
            for bc in self.data.bcs:
                if isinstance(bc, dde.icbc.PointSetBC):
                    pts_subset_idx = np.random.choice(
                        a=bc.points.shape[0],
                        size=min(bc.points.shape[0], self.num_candidates_init),
                        replace=False
                    )
                    d['bcs'].append(bc.points[jnp.array(pts_subset_idx)])
                elif isinstance(bc, dde.icbc.initial_conditions.IC):
                    d['bcs'].append(jnp.array(self.data.geom.random_initial_points(self.num_candidates_bcs)))
                else:
                    d['bcs'].append(jnp.array(self.data.geom.random_boundary_points(self.num_candidates_init)))
                # else:
                #     d['bcs'].append(jnp.array(bc.filter(self.data.train_x_bc)))
            if self.select_anchor or self.use_anc_in_train:
                d['anc'] = self.anc_point_filter(jnp.array(self.data.train_x_all))
                
        else:
            d = self.current_samples.copy()
            if (self.select_anchor and ('anc' not in d.keys())):
                d['anc'] = self.anc_point_filter(jnp.array(self.data.train_x_all))

        # Compute jacobians and eigenvalues of separate components
        def get_jacs_and_eigvals(self,d, get_eigvals=True, loss_w_bcs=None, loss_w_pde=None, loss_w_anc=None):
            loss_w_bcs = self.loss_w_bcs if loss_w_bcs is None else loss_w_bcs
            loss_w_pde = self.loss_w_pde if loss_w_pde is None else loss_w_pde
            loss_w_anc = self.loss_w_anc if loss_w_anc is None else loss_w_anc
            
            # Compute separate jacs
            jac_pde = self.ntk_fn.get_jac(d['res'], code=-1, loss_w_pde = loss_w_pde)
            jac_bcs_list = [self.ntk_fn.get_jac(d['bcs'][i], code=i, loss_w_bcs=loss_w_bcs[i]) for i in range(len(d['bcs']))] # separate eigvals for each bc component
            # jac_bcs = {k: jnp.concatenate([jc[k] for jc in jac_bcs_list], axis=0) for k in jac_pde.keys()} # for combined bc eigvals 
            
            jacs_sep = [jac_pde] + jac_bcs_list
            
            if 'anc' in d.keys():
                jac_anc = self.ntk_fn.get_jac(d['anc'], code=-2, loss_w_anc=loss_w_anc)
                jacs_sep += [jac_anc]

            # Compute eigenvalues of separate jacs
            if get_eigvals == True:
                eigvals_res = jnp.linalg.eigh(self.ntk_fn.get_ntk(jac1=jac_pde, jac2=jac_pde))[0] # second argument returns eigenvectors
                eigvals_bcs = [jnp.linalg.eigh(self.ntk_fn.get_ntk(jac1=j_, jac2=j_))[0] for j_ in jac_bcs_list]
                if 'anc' in d.keys():
                    eigvals_anc = jnp.linalg.eigh(self.ntk_fn.get_ntk(jac1=jac_anc, jac2=jac_anc))[0]
                else:
                    eigvals_anc = None
            else:
                eigvals_res = None
                eigvals_bcs = None
                eigvals_anc = None

            # Compute combined jacs
            jacs = {k: jnp.concatenate([jc[k] for jc in jacs_sep], axis=0) for k in jac_pde.keys()}

            return jacs, eigvals_res, eigvals_bcs, eigvals_anc

        def get_jac_clean_scaled(self,d, normalise_N = self.normalise_N, lr_cap = self.lr_cap, inplace = True):
        # normalise_N: normalise eigenvalues by number of points
        # lr_cap: cap learning rate at 1/lambda_max
            loss_w_pde = self.loss_w_pde
            loss_w_bcs = self.loss_w_bcs
            loss_w_anc = self.loss_w_anc
            
            has_anc = ('anc' in d.keys())
            
            # Since we are using relative scaling, we will need to set loss_w_pde to 1 here. The rest are 1 too.
            loss_w_pde = 1.0
            loss_w_bcs = [1.0 for i in range(len(d['bcs']))]
            loss_w_anc = 1.0

            _, eigvals_res, eigvals_bcs, eigvals_anc = get_jacs_and_eigvals(self,d,loss_w_pde = loss_w_pde, loss_w_bcs = loss_w_bcs, loss_w_anc=loss_w_anc, get_eigvals=True)

            # Compute number of residual and boundary points for normalising
            num_res = float(d['res'].shape[0])
            num_bcs = [float(d['bcs'][i].shape[0]) for i in range(len(d['bcs']))]
            num_anc = float(d['anc'].shape[0]) if has_anc else 0.

            # Scaling by using the max eigenvalue of the NTK matrix
            if self.scale == 'trace':
                print(f'trace_res = {jnp.sum(eigvals_res)}, trace_bcs = {[float(jnp.sum(b)) for b in eigvals_bcs]}, trace_anc = {jnp.sum(eigvals_anc) if has_anc else None}')
                print(f"Warning: full scaling including lr dependency not implemented yet.")
                loss_w_bcs = [float((jnp.sum(eigvals_res)/jnp.sum(b))**0.5) for b in eigvals_bcs]
                if has_anc:
                    loss_w_anc = float((jnp.sum(eigvals_res)/jnp.sum(eigvals_anc))**0.5)
                else:
                    loss_w_anc = self.loss_w_anc
                    
            elif self.scale == 'max':
                print(f'Before scaling: max_res = {jnp.max(eigvals_res)}, max_bcs = {[float(max(b)) for b in eigvals_bcs]}, max_anc = {jnp.max(eigvals_anc) if has_anc else None}')
                # Scaling includes normalisation by number of points
                max_eigvals_bcs = [max(eigvals_bcs[i]) for i in range(len(d['bcs']))]
                
                if normalise_N and (eigvals_anc is not None) :
                    loss_w_bcs = [float((max(eigvals_res)/max_eigvals_bcs[i])* float(num_bcs[i]/ num_res))**0.5 for i in range(len(d['bcs']))]
                else:
                    loss_w_bcs = [float((max(eigvals_res)/max_eigvals_bcs[i]))**0.5 for i in range(len(d['bcs']))]
                    
                if has_anc and (eigvals_anc is not None) :
                    if normalise_N:
                        loss_w_anc = float((max(eigvals_res)/max(eigvals_anc))* float(num_anc / num_res))**0.5
                    else:
                        loss_w_anc = float((max(eigvals_res)/max(eigvals_anc)))**0.5
                else:
                    loss_w_anc = self.loss_w_anc

            elif self.scale == 'none':
                print("No scaling applied.")
                pass
            else:
                raise ValueError(f'Invalid scale: {self.scale}')
            
            
            print(f'Normalised with N is {normalise_N} , scaling value (loss_w_bcs) = {loss_w_bcs}, scaling for anc = {loss_w_anc}. Ref scaling for res = {loss_w_pde}')

            if lr_cap:
                lambda_max = max([l * w**2 for l,w in zip(max_eigvals_bcs,loss_w_bcs)] + [max(eigvals_res)])
                if has_anc:
                    lambda_max = max(lambda_max, max(eigvals_anc))
                if self.optim_lr > 1.0/lambda_max:
                    print('Warning: lr is larger than 1/lambda_max. Rescaling loss_w_bcs and loss_w_pde')
                    normalisation_factor = 1.1 * self.optim_lr * lambda_max
                    loss_w_bcs = [float(loss_w_bcs[i]/normalisation_factor) for i in range(len(loss_w_bcs))]
                    loss_w_pde = loss_w_pde/normalisation_factor 
                    if has_anc:
                        loss_w_anc = loss_w_anc / normalisation_factor
                    # TODO to check if this is correct
                    print('New loss_w_bcs =', loss_w_bcs)
                    print('New loss_w_pde =', loss_w_pde)
                    if has_anc:
                        print('New loss_w_anc =', loss_w_anc)
                    print('max eigenvalue =', lambda_max)
                else:
                    loss_w_pde = 1.
            
            if inplace:
                print(f"Updating loss weights...")
                self.loss_w_bcs = loss_w_bcs
                self.loss_w_pde = loss_w_pde
                self.loss_w_anc = loss_w_anc
            # Now computing the combined jacobian as per usual
                jacs, eigvals_res, eigvals_bcs, eigvals_anc = get_jacs_and_eigvals(self, d)
                print('After scaling:')
                print(f'trace_res = {jnp.sum(eigvals_res)}, trace_bcs = {[float(jnp.sum(b)) for b in eigvals_bcs]}, trace_anc = {jnp.sum(eigvals_anc) if has_anc else None}')
                print(f'max_res = {jnp.max(eigvals_res)}, max_bcs = {[float(jnp.max(b)) for b in eigvals_bcs]}, max_anc = {jnp.max(eigvals_anc) if has_anc else None}')                
                return jacs
            else:
                jacs, eigvals_res, eigvals_bcs, eigvals_anc = get_jacs_and_eigvals(self,d, loss_w_pde=loss_w_pde, loss_w_bcs=loss_w_bcs, loss_w_anc=loss_w_anc)
                print('After scaling:')
                print(f'trace_res = {jnp.sum(eigvals_res)}, trace_bcs = {[float(jnp.sum(b)) for b in eigvals_bcs]}, trace_anc = {jnp.sum(eigvals_anc) if has_anc else None}')
                print(f'max_res = {jnp.max(eigvals_res)}, max_bcs = {[float(jnp.max(b)) for b in eigvals_bcs]}, max_anc = {jnp.max(eigvals_anc) if has_anc else None}')                
                return jacs, loss_w_bcs, loss_w_pde

        # Choosing the relevant jacobian computation based on whether scaling is turned on

        # TODO to optimize
        # jacs = jax.lax.cond(self.scale, get_jac_clean_scaled, get_jac_clean, (self,d)) 
        if self.scale != 'none':
            jacs = get_jac_clean_scaled(self, d, inplace=True)
        else:
            jacs = get_jacs_and_eigvals(self, d, get_eigvals=False)[0]

        # Compute NTK and get eigenvalues and eigenvectors
        # TODO to make use of only top eigenvalues
        K_train = self.ntk_fn.get_ntk(jac1=jacs, jac2=jacs)
        eigvals, eigvects = jnp.linalg.eigh(K_train + self.eps_ntk * jnp.eye(K_train.shape[0]))
        

        # ============================ Sampling for candidate points ===========================================
        # Including whole set of BC points for now
        # TODO Set how the sampling should be done. Options are 'pseudo' and 'uniform' for now
        if self.sampling == 'pseudo':
            test_pts_res = jnp.array(self.data.geom.random_points(self.num_candidates_res, random='pseudo'))
            test_pts_bc = jnp.array(self.data.geom.random_boundary_points(self.num_candidates_bcs))
            try:
                test_pts_init = jnp.array(self.data.geom.random_initial_points(self.num_candidates_init))
            except AttributeError:
                test_pts_init = None
        elif self.sampling == 'uniform':
            test_pts_res = jnp.array(self.data.geom.uniform_points(self.num_candidates_res, boundary=False))
            test_pts_bc = jnp.array(self.data.geom.uniform_boundary_points(self.num_candidates_bcs))
            try:
                test_pts_init = jnp.array(self.data.geom.uniform_initial_points(self.num_candidates_init))
            except AttributeError:
                test_pts_init = None
            
        if test_pts_init is None:
            test_pts_all = jnp.concatenate([test_pts_res, test_pts_bc], axis=0)
        else:
            test_pts_all = jnp.concatenate([test_pts_res, test_pts_bc, test_pts_init], axis=0)

        # Need to get test_pts into dictionary format
        # dict_test_pts = {
        #     'res': jnp.concatenate([jnp.array(test_pts_res), jnp.array(test_pts_bc), jnp.array(test_pts_init)], axis=0),
        #     'bcs':[
        #         jnp.array(test_pts_bc),
        #         jnp.array(test_pts_init)
        #     ],
        # }
        

        dict_test_pts = {
            'res': test_pts_res,
            'bcs': [],
        }
        
        for i, bc in enumerate(self.data.bcs):
            if isinstance(bc, dde.icbc.IC):
                # print(f'BCS_{i+1} is IC type, adding in test_pts_init')
                dict_test_pts['bcs'].append(test_pts_init)
            elif isinstance(bc, dde.icbc.PointSetBC):
                # print(f'BCS_{i+1} is PointSet type, adding in {self.num_candidates_init} candidate points from BC class')
                pts_subset_idx = np.random.choice(
                    a=bc.points.shape[0],
                    size=min(bc.points.shape[0], self.num_candidates_init),
                    replace=False
                )
                dict_test_pts['bcs'].append(jnp.array(bc.points[jnp.array(pts_subset_idx)]))
            else:
                # print(f'BCS_{i+1} is BC type, adding in test_pts_bc')
                dict_test_pts['bcs'].append(test_pts_bc)
            if self.dupl_pts_res is True:
                dict_test_pts['res']= jnp.concatenate([dict_test_pts['res'], dict_test_pts['bcs'][i]], axis=0)
                # print(f'BCS_{i+1} included in residual')
                
        if self.select_anchor:
            print(f'Anchor points included, adding in {self.num_candidates_res} candidate points')
            pts_subset_idx = np.random.choice(
                a=self.data.test_x.shape[0],
                size=min(self.data.test_x.shape[0], self.num_candidates_res),
                replace=False
            )
            dict_test_pts['anc'] = self.data.test_x[jnp.array(pts_subset_idx)]
            if self.dupl_pts_res is True:
                # print(f'Anchor points included in residual')
                dict_test_pts['res']= jnp.concatenate([dict_test_pts['res'], dict_test_pts['anc']], axis=0)


        # TODO to check whether the duplicate points for residuals are selected

        # TODO remove when boundary points are included
        if dict_test_pts['bcs'] is None:
            print("Warning: labels eig_method not implemented yet for non-empty BCs")


        # ===================== Computing the eigenvalues of the candidate K =====================

        # Computing the Jacobian of the test points
        jacs_t = get_jacs_and_eigvals(self,dict_test_pts, get_eigvals=False)[0]

        K_train_test = self.ntk_fn.get_ntk(jac1=jacs, jac2=jacs_t)

        # TODO to eventually refactor. ------------------------------
        # Note: xs cannot be a dictionary
        def _pde_residue_fn(params, xs):
            if self.inverse_problem:
                nn_params, ext_params = params
                f_ = lambda xs: self.model.net.apply(nn_params, xs, training=False)
                return self.data.pde(xs, (f_(xs), f_), ext_params)[0].reshape(-1)
            else:
                f_ = lambda xs: self.model.net.apply(params[0], xs, training=False)
                return self.data.pde(xs, (f_(xs), f_))[0].reshape(-1)

        def gen_loss_fn(params, xs, idx):
            f = generate_residue(self.data.bcs[idx], net_apply=self.model.net.apply)
            return f(params, xs)
                # return self.data.bcs[idx].error(
                #     X=xs,
                #     inputs=xs, 
                #     outputs=self.model.net.apply(params, xs, training=False), 
                #     beg=0, 
                #     end=xs.shape[0], 
                #     aux_var=None
                # ).reshape(-1)
            
        # ----------

        # Compute the MSEs of residual and boundary conditions, given input dict of test points
        def compute_residual(d):
            residual = _pde_residue_fn(self.model.params, d['res'])
            for i in range(len(d['bcs'])):
                residual = jnp.append(residual, gen_loss_fn(self.model.params[0], d['bcs'][i], idx=i))
            if 'anc' in d.keys():
                # this would need to be changed if we have multi-dim outputs
                # currently residue only just use the first dim only
                out_anc = self.model.net.apply(self.model.params[0], d['anc'], training=False)
                if self.target_fn_param is not None:
                    pseudo_target = self.model.net.apply(self.target_fn_param[0], d['anc'], training=False)
                    out_anc = out_anc - pseudo_target
                residual = jnp.append(residual, out_anc[:, self.anc_idx:self.anc_idx+1])
            return residual


        residual_train = compute_residual(d)
        residual = compute_residual(dict_test_pts)


        # ===================== Computing the scoring function P =====================
        def scoring_function(eigvects, eigvals, K_train_test, residual, P_method='labels'):
            
            idxs = eigvals > self.eps_ntk

            # print(f"Number of eigenvalues above threshold: {np.sum(idxs)}")
            # print(f"chopping indices: {idxs}")

            eigvals_chopped = eigvals[idxs]
            eigvects_chopped = eigvects[:, idxs]
            
            # Computing the scoring function P
            if P_method == 'random':
                # Without weighting by eigenvalues
                P = jnp.array(np.random.rand(K_train_test.shape[1])).reshape(1, -1)
                
            elif P_method == 'nystrom':
                # Without weighting by eigenvalues
                P = eigvects_chopped.T @ K_train_test
                
            elif P_method == 'eigvals':
                # Now trying again with weighting by eigenvalues
                P = jnp.diag(eigvals_chopped) @ eigvects_chopped.T @ K_train_test
                
            elif P_method == 'labels':
                # print(f"Residual shape: {residual.shape}")
                # print(f"train test shape: {K_train_test.shape}")
                P = jnp.diag(eigvals_chopped) @ eigvects_chopped.T @ K_train_test @ jnp.diag(residual)
                
            elif P_method == 'alignment':
                # print(f"Residual shape: {residual.shape}")
                # print(f"train test shape: {K_train_test.shape}")
                P = jnp.diag(1. / (eigvals_chopped ** 0.5)) @ eigvects_chopped.T @ K_train_test @ jnp.diag(residual)

            elif P_method == 'alignment_norm':
                # print(f"Residual shape: {residual.shape}")
                # print(f"train test shape: {K_train_test.shape}")
                P_full = jnp.diag(1. / (eigvals_chopped ** 0.5)) @ eigvects_chopped.T @ K_train_test @ jnp.diag(residual)
                P = P_full * jnp.mean(residual * (eigvects_chopped.T @ K_train_test), axis=1)[:, None]

            elif P_method == 'nystrom_wo_N':
                # P = eigvects_chopped.T @ K_train_test @ jnp.diag(residual)
                P=residual*(eigvects_chopped.T @ K_train_test)
                print(f"shape of residual is {residual.shape}, and shape of eigvects_chopped.T @ K_train_test is {(eigvects_chopped.T @ K_train_test).shape}")
                # Could try adding sqrt(n) factor, but it would not matter for ranking
                
            elif P_method == 'nystrom_norm':
                # P = eigvects_chopped.T @ K_train_test @ jnp.diag(residual)
                P_full = residual*(eigvects_chopped.T @ K_train_test)
                P = P_full * jnp.sum(P_full, axis=1)[:, None]
                print(f"shape of residual is {residual.shape}, and shape of eigvects_chopped.T @ K_train_test is {(eigvects_chopped.T @ K_train_test).shape}")

            elif P_method == 'labels_train':
                print(eigvects_chopped.shape, eigvals_chopped.shape, compute_residual(d).shape, K_train_test.shape)
                P = jnp.diag(eigvects_chopped.T @ compute_residual(d)) @ jnp.diag(eigvals_chopped) @ eigvects_chopped.T @ K_train_test
                # P = jnp.diag(eigvects_chopped.T @ compute_residual(d)) @ jnp.diag((1 - jnp.exp(-1. * eigvals_chopped)) / eigvals_chopped) @ eigvects.T @ K_train_test

            elif P_method == 'residue' or P_method == 'inverted_residue':
                P = jnp.abs(residual.reshape(1, -1))
                # P = _pde_residue_fn(self.model.net.params,dict_test_pts['res'])
                # for i in range(len(d['bcs'])):
                #     P = jnp.append(P, gen_loss_fn(self.model.net.params,dict_test_pts['bcs'][i], idx=i))
                # P = P.reshape(1, -1)
        
            return P


        def consruct_training_set_from_idx(d, idx):
            max_idx_res = d['res'].shape[0]
            # Reconstruct the flattened dic
            # test_pts_new = self.filter_dict(d,idx)
            test_pts_new = dict()
            res_idx = idx[idx < max_idx_res]
            test_pts_new['res'] = d['res'][res_idx]
            # print('res_idx =', res_idx)
            test_pts_new['bcs'] = []
            idx_start = max_idx_res
            for i, bc in enumerate(d['bcs']):
                j =  bc.shape[0]
                bc_idx = idx[(idx_start <= idx) & (idx < (idx_start + j))] - idx_start
                # print(f'bc_idx_{i} = {bc_idx}')
                bc_pts = bc[bc_idx]
                test_pts_new['bcs'].append(bc_pts)
                idx_start += j
            if self.select_anchor and ('anc' in d.keys()):
                anc_idx = idx[idx_start <= idx] - idx_start
                # print(f'anc_idx = {anc_idx}')
                test_pts_new['anc'] = d['anc'][anc_idx]
            
            # Print number of test points selected with breakdown by type res or bcs
            print(f"Number of test points selected in round: res: {test_pts_new['res'].shape[0]}, "
                  f"BCS: {[test_pts_new['bcs'][i].shape[0] for i in range(len(test_pts_new['bcs']))]}, "
                  f"Anchors: {test_pts_new['anc'].shape[0] if self.select_anchor and ('anc' in d.keys()) else '[NOT USED]'}")

            return test_pts_new
        

        # function for getting top k points based on scoring function P_rowsum
        def top_k_points(P_rowsum, d, min_num_points_res, min_num_points_bcs, num_points):
            max_idx_res = d['res'].shape[0]
            print(f"min_num_points_res is {min_num_points_res}")
            # print(f'Have {max_idx_res} res points, picking top {min_num_points_res} now.')
            _ ,idx_res = jax.lax.top_k(P_rowsum[:max_idx_res], min_num_points_res)
            remaining_budget = num_points - min_num_points_res
            idx_bcs_list = []
            idx_start = max_idx_res
            for i, bc in enumerate(d['bcs']):
                j =  bc.shape[0]
                # print(f'Have {j} bcs type {i+1} points at index {idx_start} to {idx_start + j}, picking top {self.min_num_points_bcs} now.')
                idx_bcs_list.append(jax.lax.top_k(
                    P_rowsum[idx_start:idx_start+j], 
                    min_num_points_bcs
                )[1] + idx_start)
                idx_start += j
                remaining_budget -= min_num_points_bcs
                
            
            num_anc = d['anc'].shape[0] if (self.select_anchor and ('anc' in d.keys())) else 0
            # print(f'Picking {remaining_budget - self.anchor_budget} non-anchor points now.')
            idx = jnp.concatenate([idx_res] + idx_bcs_list, axis=0)
            P_rowsum_adjusted = P_rowsum.at[idx].set(-jnp.inf)
            _, remaining_idx = jax.lax.top_k(
                P_rowsum_adjusted[:-num_anc] if (num_anc > 0) else P_rowsum_adjusted, 
                remaining_budget #- self.anchor_budget
            )
            idx = jnp.concatenate([idx, remaining_idx], axis=0)
            
            if self.select_anchor and ('anc' in d.keys()):
                # TODO: should anchor points be optional?
                print(f'Picking {self.anchor_budget} anchor points with remaining budget now.')
                P_rowsum_adjusted = P_rowsum_adjusted.at[idx].set(-jnp.inf)
                remaining_idx = jax.lax.top_k(
                    P_rowsum_adjusted[idx_start:], 
                    self.anchor_budget
                )[1] + idx_start
                idx = jnp.concatenate([idx, remaining_idx], axis=0)
            
            idx = jnp.sort(idx)
            # print(f'selected idxs = {idx}')
            return idx, consruct_training_set_from_idx(d, idx)
        
        
        def do_kmeans(P, d, min_num_points_res, min_num_points_bcs, num_points):
            P_arr = np.array(P.T)
            n = P_arr.shape[0]
            clusters = min(n, 5 * num_points)
            _, idx_ranking = kmeans_plusplus(P_arr, n_clusters=clusters)
            point_score = np.zeros(shape=(n,))
            print(f'Ranked all points with k-means, top 10 points are {idx_ranking[:10]}')
            for i, p in enumerate(idx_ranking):
                point_score[p] = float(n - i)
            return top_k_points(
                jnp.array(point_score), d,
                min_num_points_res=min_num_points_res, 
                min_num_points_bcs=min_num_points_bcs, 
                num_points=num_points,
            )
            
        def do_sampling(P, d, min_num_points_res, min_num_points_bcs, num_points):
            probs = np.array(jnp.linalg.norm(P, axis=0)**2) + 1e-9
            n = probs.shape[0]
            clusters = min(n, 5 * num_points)
            point_score = np.zeros(shape=(n,))
            idx_order = []
            for i in range(clusters):
                p = random.choices(population=list(range(n)), weights=probs)[0]
                point_score[p] = float(n - i)
                probs[p] = 0.
                idx_order.append(p)
            print(f'Ranked all points with sampling, top 10 points are {idx_order[:10]}')
            return top_k_points(
                jnp.array(point_score), d,
                min_num_points_res=min_num_points_res, 
                min_num_points_bcs=min_num_points_bcs, 
                num_points=num_points,
            )
            

        P = scoring_function(eigvects,eigvals,K_train_test,residual,self.weight_method)
        print(f"Computed the scoring function P, shape = {P.shape}")
        
        if (not self.memory) and (self.current_samples is not None) and ('anc' in self.current_samples.keys()) and (self.current_samples['anc'] is not None) and (self.current_samples['anc'].shape[0] > 0):
            num_anc = self.current_samples['anc'].shape[0]
            num_points = self.num_points_round - num_anc
        else:
            num_points = self.num_points_round

        if self.selection_method == 'greedy':
            P_rowsum = jnp.linalg.norm(P, axis=0)
            print("Selecting the top k points greedily")
            idx, test_pts_new = top_k_points(
                P_rowsum, dict_test_pts,
                min_num_points_res=self.min_num_points_res, 
                min_num_points_bcs=self.min_num_points_bcs, 
                # num_points=self.num_points_round,
                num_points=num_points,
            )
        
        elif self.selection_method == 'kmeans':
            print("Selecting the top k points using k-means++")
            idx, test_pts_new = do_kmeans(
                P, dict_test_pts,
                min_num_points_res=self.min_num_points_res, 
                min_num_points_bcs=self.min_num_points_bcs, 
                # num_points=self.num_points_round,
                num_points=num_points,
            )
            
        elif self.selection_method == 'sampling':
            print("Selecting the top k points using sampling")
            idx, test_pts_new = do_sampling(
                P, dict_test_pts,
                min_num_points_res=self.min_num_points_res, 
                min_num_points_bcs=self.min_num_points_bcs, 
                # num_points=self.num_points_round,
                num_points=num_points,
            )
        
        else:
            raise ValueError(f'Invalid selection_method {self.selection_method}')

        # to update training samples with new points. Works if init_pts is not empty
        def update_pts(init_pts, new_pts):
            returned_pts = init_pts.copy()
            returned_pts['res'] = jnp.concatenate([init_pts['res'], new_pts['res']], axis=0)
            for i in range(len(new_pts['bcs'])):
                if new_pts['bcs'][i].size != 0:
                    returned_pts['bcs'][i] = jnp.concatenate([init_pts['bcs'][i], new_pts['bcs'][i]], axis=0)
            if 'anc' in init_pts.keys() and (init_pts['anc'].shape[0] > 0) and 'anc' in new_pts.keys() and (new_pts['anc'].shape[0] > 0):
                returned_pts['anc'] = jnp.concatenate([init_pts['anc'], new_pts['anc']], axis=0)
            elif 'anc' in new_pts.keys() and (new_pts['anc'].shape[0] > 0):
                returned_pts['anc'] = new_pts['anc']
            return returned_pts


        # 'random' method to use np.random.choice to choose subset of points
        def refine_pts(d, mem_pts_ratio, enforce_budget=self.enforce_budget, include_anc_cap=self.include_anc_cap):
            print(f'## In refining ##')
            num_anc = d['anc'].shape[0] if ('anc' in d.keys()) else 0
            if enforce_budget:
                total_pts = dict_pts_size(d)
                # if include_anc_cap:
                #     if total_pts > self.mem_pts_total_budget and mem_pts_ratio < 1:
                #         # Setting mem_pts_ratio such that we stay within the budget
                #         mem_pts_ratio = (self.mem_pts_total_budget - dict_pts_size(test_pts_new)) / (total_pts)
                #         print(f"Total memory budget (exc ancs) exceeded. Setting mem_pts_ratio to {mem_pts_ratio}")
                # else:
                # if total_pts > self.mem_pts_total_budget and mem_pts_ratio < 1:
                na = test_pts_new['anc'].shape[0] if ('anc' in test_pts_new.keys()) else 0
                mem_pts_ratio = (self.mem_pts_total_budget - dict_pts_size(test_pts_new) + na) / (total_pts - num_anc)
            else:
                mem_pts_ratio = 1

            if mem_pts_ratio >= 1:
                print("Using all points as memory points")
                matrix = d
            elif self.mem_refine_selection_method == 'random':
                matrix = {
                    'res': None,
                    'bcs': []
                }
                print("Using random to choose memory points")
                idx = np.random.choice(d['res'].shape[0], int(np.floor(d['res'].shape[0]*mem_pts_ratio)), replace=False)
                matrix['res'] = d['res'][idx]
                for i in range(len(d['bcs'])):
                    if d['bcs'][i].size != 0:
                        idx = np.random.choice(d['bcs'][i].shape[0], int(np.floor(d['bcs'][i].shape[0]*mem_pts_ratio)), replace=False)
                        matrix['bcs'].append(d['bcs'][i][idx])
            # else:
            #     print(f"Using {self.mem_refine_weight_method} scoring and {self.mem_refine_selection_method} to choose memory points")
            #     # repeat K_train computation here
            #     jacs2 = get_jacs_and_eigvals(self, d, get_eigvals=False)[0]
            #     K_train_2 = self.ntk_fn.get_ntk(jac1=jacs2, jac2=jacs2)
            #     eigvals_2, eigvects_2 = jnp.linalg.eigh(K_train_2)
            #     P_rowsum_memory = scoring_function(eigvects_2,eigvals_2,K_train_2,compute_residual(d), self.mem_refine_weight_method)
            #     if self.mem_refine_selection_method == 'greedy':
            #         P_rowsum_memory = jnp.linalg.norm(P_rowsum_memory, axis=0)
            #         if self.mem_refine_weight_method == 'inverted_residue':
            #             P_rowsum_memory = - P_rowsum_memory
            #         _, matrix = top_k_points(P_rowsum_memory, d, min_num_points_res=1, min_num_points_bcs=1, num_points = (total_pts-num_anc)*mem_pts_ratio)
            #     elif self.mem_refine_selection_method == 'kmeans':
            #         _, matrix = do_kmeans(P_rowsum_memory, d, min_num_points_res=1, min_num_points_bcs=1, num_points = (total_pts-num_anc)*mem_pts_ratio)
            #     else:
            #         raise ValueError(f'Invalid mem refine method {self.mem_refine_selection_method}')
            
            if 'anc' in d.keys():
                print(f'Have {num_anc} anchors, adding them all in')
                # just keep all anchor points because it's valuable
                matrix['anc'] = d['anc']
            
            print(f"Selecting {dict_pts_size(matrix) - num_anc} points out of original {dict_pts_size(d) - num_anc} points to add to memory")
            print(f'## Out of filtering ##')

            return matrix
            

        # Adding to memory
        
        if self.memory and (dict_pts_size(test_pts_new) < self.mem_pts_total_budget):
            
            # num_anc = self.current_samples['anc'].shape[0] if ('anc' in self.current_samples.keys()) else 0

            if self.current_samples is not None:
                old_pts_sz = self.mem_pts_total_budget - dict_pts_size(test_pts_new)
                mem_pts_ratio = min(1., float(old_pts_sz) / float(dict_pts_size(self.current_samples)))
                # print(f'old pts budget = {old_pts_sz}, current sample sz (w/ anc) = {dict_pts_size(self.current_samples)} : mem_pts_ratio = {mem_pts_ratio}')

                print("Choosing points to add to memory")
                refined_old_pts = refine_pts(d=self.current_samples, mem_pts_ratio=mem_pts_ratio)
                returned_pts = update_pts(refined_old_pts, test_pts_new)
            else:
                if self.use_init_train_pts:
                    old_pts_sz = self.mem_pts_total_budget - dict_pts_size(test_pts_new)
                    mem_pts_ratio = min(1., float(old_pts_sz) / float(dict_pts_size(d)))
                    # print(f'old pts budget = {old_pts_sz}, current sample sz (w/ anc) = {dict_pts_size(d)} : mem_pts_ratio = {mem_pts_ratio}')
                    d_copy = d.copy()
                    d_copy.pop('anc')
                    refined_old_pts = refine_pts(d=d_copy, mem_pts_ratio=mem_pts_ratio)
                    returned_pts = update_pts(refined_old_pts, test_pts_new)
                
                # TODO handle exception case where only one ICBC is empty
                else: 
                    returned_pts = test_pts_new
                    
        else:
            returned_pts = test_pts_new
            if (self.current_samples is not None) and ('anc' in self.current_samples.keys()) and (self.current_samples['anc'] is not None) and (self.current_samples['anc'].shape[0] > 0):
                if ('anc' in returned_pts.keys()) and (returned_pts['anc'] is not None) and (returned_pts['anc'].shape[0] > 0):
                    returned_pts['anc'] = jnp.concatenate([self.current_samples['anc'], returned_pts['anc']], axis=0)
                else:
                    returned_pts['anc'] = self.current_samples['anc']

        # Checking if BCS is empty which shouldn't be the case since we have already set current_samples = d in the beginning
        if all(item.size==0 for item in returned_pts['bcs']) or len(returned_pts['bcs'])==0:
            print("Warning: There are no BCs selected in test points. Adding back all original BCs")
            returned_pts['bcs']=[jnp.array(bc.filter(self.data.train_x_bc)) for bc in self.data.bcs] #using the whole bc dataset

        # Print number of test points selected with breakdown by type res or bcs
        print(f"Number of test points in total: res: {returned_pts['res'].shape[0]}, "
              f"BCS: {[returned_pts['bcs'][i].shape[0] for i in range(len(returned_pts['bcs']))]}, "
              f"Anchors: {returned_pts['anc'].shape[0] if 'anc' in returned_pts.keys() else 0}")

        if self.scale!='none':
            print('---------------------\nScaling for training:')
            _  = get_jac_clean_scaled(self,returned_pts, inplace=True) # self.loss_w_bcs and self.loss_w_pde are updated in place

        # Compute projection of labels and eigvectors
        def compute_projection(new_pts):
            jacs_new_pts = get_jacs_and_eigvals(self, new_pts, get_eigvals=False)[0]
            K_train_new_pts = self.ntk_fn.get_ntk(jac1=jacs, jac2=jacs_new_pts)
            # print(f"K_train_new_pts shape is {K_train_new_pts.shape}")
            # print(f"compute_residual(new_pts) shape is {compute_residual(new_pts).shape}")
            # print(f"eigvects.T shape is {eigvects.T.shape}")
            a = (eigvects.T @ K_train_new_pts)@ compute_residual(new_pts)
            a_top, a_idx = jax.lax.top_k(a,15)
            label_info_new_pts = {
                'a':a,
                'a_top':a_top,
                'a_idx':a_idx,
                'a_norm':jnp.linalg.norm(a),
            }
            return label_info_new_pts

        # Calculating this for new points and returned points
        label_info_new_pts = compute_projection(test_pts_new)
        # print(f"len of a_idx is {len(label_info_new_pts['a_idx'])}, and len of a_top is {len(label_info_new_pts['a_top'])}")
        # print(f"Top 15 coeff and corresponding eigvectors index for new points: {label_info_new_pts['a_idx']} out of {len(label_info_new_pts['a'])}, values are: {label_info_new_pts['a_top']}")
        label_info_returned_pts = compute_projection(returned_pts)
        # print(f"Top 15 coeff and corresponding eigvectors index for returned points: {label_info_returned_pts['a_idx']} out of {len(label_info_returned_pts['a'])}, values are: {label_info_returned_pts['a_top']}")

        # Return dictionary of all relevant data for logging
        logging_dict = {
            'chosen_pts': returned_pts,
            'eigvals': eigvals,
            'eigvects': eigvects,
            'P': P,
            'jac_train': jacs,
            'jac_candidates': jacs_t,
            'K_train_test': K_train_test,
            'NTK': K_train,
            'candidate_pts': dict_test_pts,
            'selected_pts_idx': idx,
            'old_points': d,
            'new_points': test_pts_new,
            'residual_old': residual_train,
            'residual_candidates': residual,
            'new_loss_w_bcs': self.loss_w_bcs,
            'new_loss_w_pde': self.loss_w_pde,
            'new_loss_w_anc': self.loss_w_anc,
            'label_info_new_pts': label_info_new_pts,
            'label_info_returned_pts': label_info_returned_pts,
        }
        
        for k in logging_dict:
            logging_dict[k] = to_cpu(logging_dict[k])

        return returned_pts, logging_dict




