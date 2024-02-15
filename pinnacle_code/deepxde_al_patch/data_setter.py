import numpy as np

from . import deepxde as dde
from .deepxde import config


# class DataSetterCallback(dde.callbacks.Callback):
    
#     def __init__(self):
#         super().__init__(self)
#         self.xs = None
    
#     def set_data(self, xs):
#         self.xs = xs
    
#     def on_train_begin(self):
#         soln = self.model.data.soln
#         ys = soln(self.xs) if soln else None
#         self.train_state.set_data_train(X_train=self.xs, y_train=ys)


def _is_iterable(obj):
    try:
        some_object_iterator = iter(obj)
        return True
    except TypeError as te:
        return False


class DataSetter:
    
    def __init__(self, model: dde.Model):
        self.model = model
        self.net = model.net
        self.pde_data = model.data
    
    def replace_points(self, train_pts):
        pd = self.pde_data
        
        pd.train_x_all = None
        pd.train_x_bc = None
        pd.train_x, pd.train_y, pd.train_aux_vars = None, None, None
        pd.anchors = None
        
        # X = np.empty((0, pd.geom.dim), dtype=config.real(np))
        # X = np.vstack([np.array(x) for x in train_pts['bcs']] + [np.array(train_pts['res'])])
        # if self.num_boundary > 0:
        #     if self.train_distribution == "uniform":
        #         tmp = self.geom.uniform_boundary_points(self.num_boundary)
        #     else:
        #         tmp = self.geom.random_boundary_points(
        #             self.num_boundary, random=self.train_distribution
        #         )
        #     X = np.vstack((tmp, X))
        # if self.anchors is not None:
        #     X = np.vstack((self.anchors, X))
        # if self.exclusions is not None:

        #     def is_not_excluded(x):
        #         return not np.any([np.allclose(x, y) for y in self.exclusions])

        #     X = np.array(list(filter(is_not_excluded, X)))
        
        # pd.train_x_all = np.vstack([np.array(x) for x in train_pts['bcs']] + [np.array(train_pts['res'])])
        pd.train_x_all = np.array(train_pts['res'])
        
        # pd.bc_points()  # Generate self.num_bcs and self.train_x_bc
        bcs_pts = [np.array(x) for x in train_pts['bcs']]
        pd.num_bcs = list(map(len, bcs_pts))
        pd.train_x_bc = np.vstack(bcs_pts)
        
        # pd.train_x_all = np.array(train_pts['res'])
        if pd.pde is not None:
            pd.train_x = np.vstack((pd.train_x_bc, pd.train_x_all))
        else:
            pd.train_x = pd.train_x_bc
            
        pd.train_y = pd.soln(pd.train_x) if pd.soln else None
        
        if pd.auxiliary_var_fn is not None:
            pd.train_aux_vars = pd.auxiliary_var_fn(pd.train_x).astype(config.real(np))
            
        return pd.train_x, pd.train_y, pd.train_aux_vars
