from functools import partial
import os
from collections.abc import MutableMapping

import jax
import jax.numpy as jnp
import flax

from . import deepxde as dde

from .icbc_patch import generate_residue


def _flatten_dict(d, parent_key='', sep='_'):
    # https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    # needed as neural network parameters are stored in a nested dictionary, e.g. {'params': {'dense': {'kernel': ...}}}
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping) or isinstance(v, flax.core.frozen_dict.FrozenDict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# @partial(jax.jit, static_argnames=['net_fn', 'der_fn'])
# # The reason for using partial(jax.jit, static_argnames=['net_fn', 'der_fn']) is that the function _jac_params_helper() is called in a loop in the function _jac_params() below. If we do not use partial(jax.jit, static_argnames=['net_fn', 'der_fn']), the function _jac_params_helper() will be recompiled in each iteration of the loop, which is very inefficient. @partial basically fixes the arguments net_fn and der_fn, so that the function _jac_params_helper() is only compiled once.
# def _jac_params_helper(params, x, net_fn, der_fn):
#     # params is a dictionary of parameters of the neural network
#     # net_fn is a function that takes in params and x, and returns the output of the neural network
#     # der_fn is a function that takes in x and the output of the neural network, and returns the derivative of the output of the neural network with respect to x
#     def f2_(params):
#         f_ = lambda xs: net_fn(params, xs)
#         return der_fn(x, (f_(x), f_))
#     # j_ is the Jacobian of f2_ with respect to params
#     # dd is a dictionary of Jacobians of f2_ with respect to each parameter in params.
#     j_ = lambda params: jax.jacrev(fun=f2_, has_aux=False)(params)
#     dd = j_(params)
#     return dd[0]

# @partial(jax.jit, static_argnames=['fn'])
# def _jac_params_helper(params, x, fn):
#     j_ = lambda params: jax.jacobian(fun=fn, has_aux=False)(params, x)
#     dd = j_(params)
#     return dd

@partial(jax.jit, static_argnames=['fn'])
def _jac_params_helper(params, x, fn):
    fn2 = lambda params, x_: fn(params, x_.reshape(1, -1))[0]  # version for single dims
    # print(fn2(params, x).shape)
    f_ = lambda x_: jax.jacobian(fun=fn2, has_aux=False)(params, x_)
    dd = jax.vmap(f_)(x)
    # dd = jax.jit(jax.vmap(jax.grad(fn2), in_axes=(None, 0)))(params, x)
    return dd


def _jac_params_cleanup(dd):
    dd = _flatten_dict(dd['params'])
    # currently only works for one-dimensional model outputs
    return {k: dd[k].reshape(dd[k].shape[0], -1) for k in dd.keys()}


def get_ntk_from_jac(jac1, jac2):
    # prods = [jnp.einsum('ijk,ljk->jil', jac1[k], jac2[k]) for k in jac1.keys()]
    # return sum(prods)
    prods = None
    for k in jac1.keys():
        m = jac1[k] @ jac2[k].T
        prods = m if (prods is None) else (prods + m)
    return prods


class NTKHelper:
    
    def __init__(self, model: dde.Model, inverse_problem: bool = False):
        self.model = model
        self.inverse_problem = inverse_problem
        self.net = model.net
        self.pde = model.data.pde
        self.bcs = model.data.bcs
        self.bc_fns = [generate_residue(bc, self.net.apply, return_output_for_pointset=True) for bc in self.bcs]
        self._output_fn = lambda params, xs: self.net.apply(params, xs, training=True)
    
    def get_jac_clean(self,d,loss_w_bcs=1.0, loss_w_pde=1.0):
        jac_pde = self.get_jac(d['res'], code=-1, loss_w_pde=loss_w_pde)
        jac_bcs = [self.get_jac(d['bcs'][i], code=i, loss_w_bcs = loss_w_bcs) for i in range(len(d['bcs']))]
        jacs_sep = [jac_pde] + jac_bcs
        jacs = {k: jnp.concatenate([jc[k] for jc in jacs_sep], axis=0) for k in jac_pde.keys()}
        return jacs

    def _get_output_jac(self, xs, params, loss_w_anc=1.0):
        fun=lambda params, x: loss_w_anc * self._output_fn(params, x)[:, 0]
        d = _jac_params_helper(params=params, x=xs, fn=fun)
        return _jac_params_cleanup(d)
        
    def _get_pde_jac(self, xs, params, loss_w_pde = 1.0):
        
        def f2_(params, x):
            f_ = lambda x: self.net.apply(params, x)
            if self.inverse_problem:
                return loss_w_pde * self.pde(x, (f_(x), f_), self.model.params[1])[0]
            else:
                return loss_w_pde * self.pde(x, (f_(x), f_))[0]
        
        d = _jac_params_helper(params=params, x=xs, fn=f2_)
        return _jac_params_cleanup(d)
    
    def get_pde_jac_inv(self, xs, params):
        
        def res_fn(params, x):
            nn_param, pde_param = params
            f_ = lambda x: self.net.apply(nn_param, x)
            return self.pde(x, (f_(x), f_), pde_param)[0]
        
        d = _jac_params_helper(params=params, x=xs, fn=res_fn)
        d[0] = _jac_params_cleanup(d[0])
        return d
    
    def get_pde_jac_crossterm(self, xs, params):
        cross_term = jax.jacfwd(lambda inv_param, nn_param, xs: self.get_pde_jac_inv(xs, [nn_param, inv_param])[0])
        jac_cross = cross_term(params[1], params[0], xs)
        return [{k: jac_cross[k][i] for k in jac_cross.keys()} for i in range(len(params[1]))]
    
    
    # Adding loss_w_bcs to introduce loss weights
    def _get_bc_jac(self, bc_idx, xs, params, loss_w_bcs = 1.0):
        # bc = self.bcs[bc_idx]         
        # # bc_fn = lambda xs, ys: (loss_w_bcs* bc.error(xs, xs, ys[0], 0, xs.shape[0]),)
        # bc_fn = lambda xs, ys: (loss_w_bcs* bc.error(xs, xs, ys[0], 0, xs.shape[0]),)
        d = _jac_params_helper(
            params=params, x=xs, 
            fn=lambda param, x_: loss_w_bcs * self.bc_fns[bc_idx](param, x_)
        )
        return _jac_params_cleanup(d)
    
    def _get_jac_fn(self, code, params=None, loss_w_bcs=1.0, loss_w_pde=1.0, loss_w_anc=1.0):
        if params is None:
            # use param as stored in self.net
            # this disregards the effect that comes from ext_params tho
            params = self.net.params
            
        if code == -2:
            # derivative wrt output only
            return partial(self._get_output_jac, params=params, loss_w_anc=loss_w_anc)
        elif code == -1:
            # derivative wrt PDE residue
            return partial(self._get_pde_jac, params=params, loss_w_pde=loss_w_pde)
        else:
            # derivative wrt BC error term
            assert 0 <= code < len(self.bcs)
            return partial(self._get_bc_jac, bc_idx=code, params=params, loss_w_bcs=loss_w_bcs)
        
    def get_jac(self, xs, code=-2, params=None, loss_w_bcs = 1.0, loss_w_pde = 1.0, loss_w_anc = 1.0):
        return self._get_jac_fn(code=code, params=params, loss_w_bcs=loss_w_bcs, loss_w_pde=loss_w_pde,loss_w_anc=loss_w_anc)(xs=xs)
    
    def get_ntk(self, xs1=None, code1=-2, jac1=None, xs2=None, code2=None, jac2=None, params=None):
        """compute the (empirical) NTK between two inputs under some transformation

        Parameters
        ----------
        xs1 : jax Array
            array 1
        code1 : int, optional
            derivative wrt function output (-2), PDE residual (-1) or BC error (non-neg int), by default -2
        xs2 : jax Array
            array 2, by default None
        code2 : int, optional
            derivative wrt function output (-2), PDE residual (-1) or BC error (non-neg int), 
            by default None (use same as code1)
        params : FrozenDict, optional
            parameter of net, by default None

        Returns
        -------
        jax Array
            empirical NTK with dimension
            (net_output.shape[1], xs1.shape[0], xs2.shape[0])
        """
        if jac1 is None:
            assert xs1 is not None
            jac1 = self.get_jac(xs=xs1, code=code1, params=params)
            print(f"Warning: jac1 not provided so computing jac without custom loss parameters")
        
        if jac2 is not None:
            # jac2 already specified so use it
            pass
        elif xs2 is None:
            # otherwise if not specified, and xs2 not given, assume xs2 = xs1 with same codes
            jac2 = jac1
            print(f"Warning: jac2 not provided so computing jac without custom loss parameters")
        else:
            # otherwise compute the jacobian for xs2
            code2 = code1 if code2 is None else code2
            print(f"Warning: jac2 not provided so computing jac without custom loss parameters")
            jac2 = self._get_jac_fn(code=code2, params=params)(xs=xs2)
        
        return get_ntk_from_jac(jac1=jac1, jac2=jac2)
