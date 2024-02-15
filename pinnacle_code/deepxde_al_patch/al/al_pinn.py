from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

from .. import deepxde as dde

from ..data_setter import DataSetter


class PointSelector:
    
    def __init__(self, model: dde.Model, inverse_problem: bool = False,
                 current_samples: dict = None, anchor_budget: int = 0, anc_point_filter: Callable = None, anc_idx: int = None,
                 mem_pts_total_budget: int  = 1000, min_num_points_bcs: int = 0, min_num_points_res: int = 0,
                 loss_w_bcs: float = 1., loss_w_pde: float = 1., loss_w_anc: float = 1., optim_lr: float = 1e-3, enforce_budget: bool = True):
        self.model = model
        self.inverse_problem = inverse_problem
        self.current_samples = current_samples
        self.anchor_budget = anchor_budget
        self.anc_point_filter = anc_point_filter if anc_point_filter else (lambda xs: xs)
        self.anc_idx = anc_idx
        self.mem_pts_total_budget = mem_pts_total_budget
        self.enforce_budget = enforce_budget #(mem_pts_total_budget is not None)
        self.min_num_points_bcs = min_num_points_bcs
        self.min_num_points_res = min_num_points_res
        self.loss_w_bcs = loss_w_bcs if hasattr(loss_w_bcs, "__len__") else [loss_w_bcs for _ in self.data.bcs]
        self.loss_w_pde = loss_w_pde
        self.loss_w_anc = loss_w_anc
        self.optim_lr = optim_lr
        
        self.select_anchor = (anchor_budget > 0)
        self.data = self.model.data
        self.bcs = self.model.data.bcs
        self._last_train_pts = None
        
    def generate_samples(self):
        raise NotImplementedError
    
    def set_data(self, train_pts=None):
        train_pts = train_pts if train_pts else self._last_train_pts
        DataSetter(model=self.model).replace_points(train_pts=train_pts)
