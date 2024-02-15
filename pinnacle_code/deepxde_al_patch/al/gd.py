# from functools import partial
# import os
# import pickle as pkl
# from collections.abc import MutableMapping

# import matplotlib.pyplot as plt
# import numpy as np
# import tqdm

# import jax
# import jax.numpy as jnp
# import flax
# from flax import linen as nn
# import optax

# from .. import deepxde as dde

# from ..ntk import NTKHelper
# from ..icbc_patch import generate_residue, constrain_domain, constrain_ic, constrain_bc
# from ..utils import pairwise_dist
# from .ntk_based_al import NTKBasedAL


# class GDPointSelector(NTKBasedAL):
    
#     def __init__(self, model: dde.Model, 
#                  inverse_problem: bool = False,
#                  current_samples: dict = None,
#                  weight_method: str = "labels", # possible options are 'none', 'labels', 'eigvals', 'labels_train'
#                  mem_pts_total_budget: int = None, # total number of points to select
#                  min_num_points_bcs: int = 0,
#                  min_num_points_res: int = 0,
#                  loss_w_bcs: float = [1.0, 1.0],  
#                  loss_w_pde: float = 1.0, 
#                  loss_w_anc: float = 1.0,
#                  anchor_budget: int = 0,
#                  anc_point_filter=None,
#                  optim_lr : float = 1e-3,
#                  points_pool_size: int = 1000, 
#                  active_eig: int = None,
#                  eig_min: float = 1e-2, 
#                  num_points_round: int = 30,
#                  eps: float = 1e-6, 
#                  lr: float = 1e-2, 
#                  train_steps: int = 1000, 
#                  indicator: str = 'K', 
#                  compare_mode: bool = True, 
#                  crit: str = 'fr',
#                  reinit_from_sample: bool = True, 
#                  scale: str = None,
#                  decay_factor: float = 1.,
#                  dist_reg: float = 0.):
#         super().__init__(
#             model=model, points_pool_size=points_pool_size, eig_min=eig_min, active_eig=active_eig,
#             inverse_problem=inverse_problem, current_samples=current_samples, anchor_budget=anchor_budget, anc_point_filter=anc_point_filter,
#             mem_pts_total_budget=mem_pts_total_budget, min_num_points_bcs=min_num_points_bcs, min_num_points_res=min_num_points_res, 
#             loss_w_anc=loss_w_anc, loss_w_bcs=loss_w_bcs, loss_w_pde=loss_w_pde, optim_lr=optim_lr
#         )
#         self.num_points_round = num_points_round
#         self.current_samples = current_samples
#         self.loss_w_bcs = loss_w_bcs
#         self.eps = eps
#         self.lr = lr
#         self.train_steps = train_steps
#         self.crit = crit
#         self.compare_mode = compare_mode
#         self.indicator = indicator
#         self.reinit_from_sample = reinit_from_sample
#         self.scale = scale
#         self.decay_factor = decay_factor
#         self.dist_reg = dist_reg
        
#     def generate_samples(self):
        
#         if self.current_samples is None:
#             nt = self.points_pool_size // 3
#             test_pts_res = jnp.array(self.data.geom.random_points(nt, random='pseudo'))
#             test_pts_bc = jnp.array(self.data.geom.random_boundary_points(nt))
#             test_pts_init = jnp.array(self.data.geom.random_initial_points(nt))
#             test_pts_all = jnp.concatenate([test_pts_res, test_pts_bc, test_pts_init], axis=0)
#             dtest = {
#                 'res': test_pts_all,
#                 'bcs': [jnp.array(bc.filter(test_pts_all)) for bc in self.data.bcs]
#             }
#         else:
#             dtest = self.current_samples
        
#         jac_pde = self.ntk_fn.get_jac(dtest['res'], code=-1)
#         # to refactor
#         jac_bcs_list = [self.ntk_fn.get_jac(dtest['bcs'][i], code=i, loss_w_bcs=1.) for i in range(len(dtest['bcs']))]
#         jac_bcs = {k: jnp.concatenate([jc[k] for jc in jac_bcs_list], axis=0) for k in jac_pde.keys()}

#         # Compute NTK for residual points and get eigenvalues and eigenvectors
#         T_t_res = self.ntk_fn.get_ntk(jac1=jac_pde, jac2=jac_pde)
#         eigvals_res = jnp.linalg.eigh(T_t_res)[0] / float(dtest['res'].shape[0])

#         # # Compute NTK for BC points and get eigenvalues and eigenvectors
#         # T_t_bcs = self.ntk_fn.get_ntk(jac1=jac_bcs, jac2=jac_bcs)
#         # eigvals_bcs, _  = jnp.linalg.eigh(T_t_bcs)
        
#         eigvals_bcs = [jnp.linalg.eigh(self.ntk_fn.get_ntk(jac1=j_))[0] / float(xb.shape[0])
#                        for (xb, j_) in zip(dtest['bcs'], jac_bcs_list)]

#         # Scaling by using the max eigenvalue of the NTK matrix
#         # if self.scale == 'trace':
#         #     print(f'trace_res = {jnp.sum(eigvals_res)}, trace_bcs = {[float(jnp.sum(b)) for b in eigvals_bcs]}')
#         #     loss_w_bcs = [float((jnp.sum(eigvals_res)/jnp.sum(b))**0.5) for b in eigvals_bcs]
#         # elif self.scale == 'max':
#         print('To figure out allocation of points:')
#         print(f'Matrix size: res = {eigvals_res.shape[0]}, bcs = {[b.shape[0] for b in eigvals_bcs]}')
#         print(f'Max eigval: res = {jnp.max(eigvals_res)}, bcs = {[float(max(b)) for b in eigvals_bcs]}')
#         loss_w_bcs = [float(max(b) / max(eigvals_res)) ** 0.5 for b in eigvals_bcs]
#         # else:
#         #     raise ValueError(f'Invalid scale: {self.scale}')
        
#         min_points_used = self.min_num_points_res + (self.min_num_points_bcs * len(loss_w_bcs))
#         factor = float(self.num_points_round - min_points_used) / (1. + sum(loss_w_bcs))
#         n_bcs = [max(1, int(b * factor)) + self.min_num_points_bcs for b in loss_w_bcs]
#         n_res = self.num_points_round - sum(n_bcs)
#         print(f'n_res = {n_res}, n_bcs = {n_bcs}')
            
#         if self.indicator == 'Kt':
            
#             # pred_mat = lambda d: self.get_K_subset(d)[0]
#             def pred_mat(d):
#                 jac_pde = self.ntk_fn.get_jac(d['res'], code=-1)
#                 jac_bcs = [self.ntk_fn.get_jac(d['bcs'][i], code=i) for i in range(len(d['bcs']))]
                
#                 jacs_sep = [jac_pde] + jac_bcs
#                 jacs = {k: jnp.concatenate([jc[k] for jc in jacs_sep], axis=0) for k in jac_pde.keys()}
                            
#                 T_t = self.ntk_fn.get_ntk(jac1=jacs, jac2=jacs)
#                 T_t = T_t + self.eps * jnp.eye(T_t.shape[0])
#                 eigvals, eigvects = jnp.linalg.eigh(T_t)
#                 eigvals_sub = eigvals[-self.active_eig:]
#                 eigvects_sub = eigvects[:,-self.active_eig:]
#                 T_t_inv = eigvects_sub @ jnp.diag(1. / eigvals_sub) @ eigvects_sub.T
#                 # T_t_inv = jnp.linalg.inv(T_t)
                
#                 T_nt = self.ntk_fn.get_ntk(jac1=self.jac_all, jac2=jacs)
#                 K_subset = T_nt @ T_t_inv @ T_nt.T
                
#                 return K_subset
            

#             target_mat = self.K_reducedrank
            
#         elif self.indicator == 'span':
            
#             assert self.crit in {'fr'}  # only Frobenius norm makes sense here
#             # assert not self.compare_mode  # TODO: compare with if we train with whole dataset
#             _K = lambda d: self.get_K_subset(d)[0]
#             f_ = lambda xs: self.model.net.apply(self.model.net.params, xs, training=False)
        
#             def gen_loss_fn(params, xs, idx):
#                 f = generate_residue(self.data.bcs[idx], net_apply=self.model.net.apply)
#                 return f(params, xs)
        
#             def _generate_pseudolabels(d):
#                 _pde_comp = self.data.pde(d['res'], (f_(d['res']), f_))[0].reshape(-1)
#                 # _bcs_comp = [self.loss_w_bcs * bc.error(
#                 #     X=xs, 
#                 #     inputs=xs, 
#                 #     outputs=f_(xs), 
#                 #     beg=0, 
#                 #     end=xs.shape[0], 
#                 #     aux_var=None
#                 # ).reshape(-1, 1) for bc, xs in zip(self.data.bcs, d['bcs'])]
#                 _bcs_comp = [gen_loss_fn(self.model.net.params, d['bcs'][i], idx=i).reshape(-1)
#                              for i in range(len(self.data.bcs))]
#                 return jnp.concatenate([_pde_comp] + _bcs_comp, axis=0)
            
#             def pred_mat(d):
#                 plabels = _generate_pseudolabels(d)
                
#                 jac_pde = self.ntk_fn.get_jac(d['res'], code=-1)
#                 jac_bcs = [self.ntk_fn.get_jac(d['bcs'][i], code=i) for i in range(len(d['bcs']))]
                
#                 jacs_sep = [jac_pde] + jac_bcs
#                 jacs = {k: jnp.concatenate([jc[k] for jc in jacs_sep], axis=0) for k in jac_pde.keys()}
                            
#                 T_t = self.ntk_fn.get_ntk(jac1=jacs, jac2=jacs)
#                 T_t = T_t + self.eps * jnp.eye(T_t.shape[0])
#                 eigvals, eigvects = jnp.linalg.eigh(T_t)
#                 eigvects_sub = eigvects
#                 eigvals_sub = (1 - jnp.exp(-self.decay_factor * eigvals)) / eigvals
#                 # eigvals_sub = eigvals[-self.active_eig:]
#                 # eigvects_sub = eigvects[:,-self.active_eig:]
#                 # T_t_inv = eigvects_sub @ jnp.diag(1. / eigvals_sub) @ eigvects_sub.T
#                 # T_t_inv = jnp.linalg.inv(T_t)
                
#                 T_nt = self.ntk_fn.get_ntk(jac1=self.jac_all, jac2=jacs)
                                
#                 # return T_nt @ T_t_inv @ plabels
#                 return (eigvects_sub.T @ plabels) @ (jnp.diag(eigvals_sub) @ eigvects_sub.T) @ T_nt.T
            
#             nt = self.points_pool_size // 3
#             test_pts_res = jnp.array(self.data.geom.random_points(nt, random='pseudo'))
#             test_pts_bc = jnp.array(self.data.geom.random_boundary_points(nt))
#             test_pts_init = jnp.array(self.data.geom.random_initial_points(nt))
#             test_pts_all = jnp.concatenate([test_pts_res, test_pts_bc, test_pts_init], axis=0)
#             dtest = {
#                 'res': test_pts_all,
#                 'bcs': [jnp.array(bc.filter(test_pts_all)) for bc in self.data.bcs]
#             }
#             target_mat = pred_mat(dtest)
            
#         if self.compare_mode:   
#             if self.crit == 'sp':
#                 def score_crit(d):
#                     return jnp.max(jnp.linalg.eigvalsh(pred_mat(d) - target_mat))**2
#             elif self.crit == 'fr':
#                 def score_crit(d):
#                     return jnp.mean((pred_mat(d) - target_mat)**2)
#             else:
#                 raise ValueError(f'Invalid crit {self.crit}')
            
#         else:
#             if self.crit == 'sp':
#                 def score_crit(d):
#                     return -jnp.max(jnp.linalg.eigvalsh(pred_mat(d)))**2
#             elif self.crit == 'fr':
#                 def score_crit(d):
#                     return -jnp.mean((pred_mat(d))**2)
#             else:
#                 raise ValueError(f'Invalid crit {self.crit}')
            
#         if self.dist_reg > 0:
#             def score(d):
#                 crit_score = score_crit(d)
#                 dist_res = jnp.mean(pairwise_dist(d['res'], d['res']))
#                 dist_bcs = [jnp.mean(pairwise_dist(b, b)) for b in d['bcs']]
#                 avg_dist = (dist_res + sum(dist_bcs)) / (1. + len(dist_bcs))
#                 return crit_score + self.dist_reg * avg_dist
#         else:
#             score = score_crit
        
#         # score = score_dict[self.crit]
#         val_grad_fn = jax.jit(jax.value_and_grad(score))
#         # val_grad_fn = jax.value_and_grad(score)
        
#         # if self.reinit_from_sample or (self.current_samples is None):
#         train_pts = self.constrain({
#             'res':  jnp.array(self.data.geom.random_points(n_res)),
#             'bcs': [jnp.array(self.data.geom.random_points(b)) for b in n_bcs]
#         })
#         # else:
#         #     train_pts = self.constrain(self.current_samples)
        
#         opt = optax.adam(learning_rate=self.lr)
#         opt_state = opt.init(train_pts)
#         self._last_train_scores = []
#         print(f'pred_mat shape = {pred_mat(train_pts).shape}')

#         for i in tqdm.trange(self.train_steps):
#             val, grad = val_grad_fn(train_pts)
#             self._last_train_scores.append(val)
#             updates, opt_state = opt.update(grad, opt_state)
#             train_pts_ = optax.apply_updates(train_pts, updates)
#             if jnp.isnan(train_pts_['res']).any() or np.any([jnp.isnan(b).any() for b in train_pts_['bcs']]):
#                 print('ENCOUNTERED nan, ABORTING...')
#                 # print(jnp.isnan(train_pts_['res']), [jnp.isnan(b).any() for b in train_pts_['bcs']])
#                 # print(train_pts_)
#                 break
#             train_pts = train_pts_
#             train_pts = self.constrain(train_pts)
#             self._last_train_pts = train_pts
            
#         self._K_subset = self.get_K_subset(train_pts)
#         round_selected_samples = train_pts.copy()
        
#         if self.current_samples is not None:
            
#             current_samples_allowed = self.mem_pts_total_budget - self.num_points_round
#             n_current = self.current_samples['res'].shape[0] + sum([b.shape[0] for b in self.current_samples['bcs']])
#             if n_current < current_samples_allowed:
#                 print(f'current_sample size = {n_current}, mem_allowed = {current_samples_allowed} - keep all current_samples')
#                 s = self.current_samples.copy()
#             else:
#                 keep_factor = current_samples_allowed / n_current
#                 print(f'current_sample size = {n_current}, mem_allowed = {current_samples_allowed} - keep only {keep_factor:.3f} of current_samples')
#                 res_keep = jnp.array(np.random.choice(
#                     a=self.current_samples['res'].shape[0],
#                     size=int(keep_factor * self.current_samples['res'].shape[0]),
#                     replace=False
#                 ))
#                 bcs_keep = [jnp.array(np.random.choice(
#                     a=b.shape[0],
#                     size=int(keep_factor * b.shape[0]),
#                     replace=False
#                 )) for b in self.current_samples['bcs']]
#                 s = {
#                     'res': self.current_samples['res'][res_keep, :],
#                     'bcs': [b[bk, :] for b, bk in zip(self.current_samples['bcs'], bcs_keep)]
#                 }
#             print(f'reduced cur_sample size: res = {s["res"].shape[0]}, bcs = {[b.shape[0] for b in s["bcs"]]}')
            
#             train_pts = {
#                 'res': jnp.concatenate([s['res'], train_pts['res']]),
#                 'bcs': [jnp.concatenate([bc1, bc2]) for (bc1, bc2) in zip(s['bcs'], train_pts['bcs'])]
#             }
#             print(f'final sample size: res = {train_pts["res"].shape[0]}, bcs = {[b.shape[0] for b in train_pts["bcs"]]}')
             
#         # if self.scale:
#         #     print('---------------------\nScaling for training:')
#         #     d = train_pts
            
#         #     jac_pde = self.ntk_fn.get_jac(d['res'], code=-1)
#         #     # to refactor
#         #     jac_bcs_list = [self.ntk_fn.get_jac(d['bcs'][i], code=i, loss_w_bcs=1.) for i in range(len(d['bcs']))]
#         #     jac_bcs = {k: jnp.concatenate([jc[k] for jc in jac_bcs_list], axis=0) for k in jac_pde.keys()}

#         #     # Compute NTK for residual points and get eigenvalues and eigenvectors
#         #     T_t_res = self.ntk_fn.get_ntk(jac1=jac_pde, jac2=jac_pde)
#         #     eigvals_res, _ = jnp.linalg.eigh(T_t_res)

#         #     # Compute NTK for BC points and get eigenvalues and eigenvectors
#         #     T_t_bcs = self.ntk_fn.get_ntk(jac1=jac_bcs, jac2=jac_bcs)
#         #     eigvals_bcs, _  = jnp.linalg.eigh(T_t_bcs)

#         #     # Scaling by using the max eigenvalue of the NTK matrix
#         #     if self.scale == 'trace':
#         #         print(f'trace_res = {jnp.sum(eigvals_res)}, trace_bcs = {jnp.sum(eigvals_bcs)}')
#         #         scale_term = float((jnp.sum(eigvals_res)/jnp.sum(eigvals_bcs))**0.5)
#         #     elif self.scale == 'max':
#         #         print(f'max_res = {jnp.max(eigvals_res)}, max_bcs = {jnp.max(eigvals_bcs)}')
#         #         scale_term = float((max(eigvals_res)/max(eigvals_bcs))**0.5)
#         #     else:
#         #         raise ValueError(f'Invalid scale: {self.scale}')
            
#         #     num_res = float(d['res'].shape[0])
#         #     num_bcs = float(sum([d['bcs'][i].shape[0] for i in range(len(d['bcs']))]))
#         #     print(f'num_res = {num_res}, num_bcs = {num_bcs}')
#         #     self.loss_w_bcs = scale_term * float((num_bcs / num_res) ** 0.5)
            
#         #     print(f'loss_w_bcs = {self.loss_w_bcs}')
                
#         intermediates = {
#             'indicator': self.indicator,
#             'compare_mode': self.compare_mode,
#             'crit': self.crit,
#             'jac_all': self.jac_all,
#             'K_fullrank': self.K_fullrank,
#             'K_reducedrank': self.K_reducedrank,
#             'active_eig': self.active_eig,
#             'train_pts': train_pts,
#             'round_selected_samples': round_selected_samples,
#             'K_subset_data': self._K_subset,
#             'scores': self._last_train_scores,
#             'new_loss_w_bcs': self.loss_w_bcs,
#         }
            
#         return train_pts, intermediates
