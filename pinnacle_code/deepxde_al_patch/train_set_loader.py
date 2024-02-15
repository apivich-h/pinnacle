import os

import numpy as np
import jax
import jax.numpy as jnp

from . import deepxde as dde

from .pdebench_pinn.train import *


def load_data(pde_name, pde_const=tuple(), use_pdebench=False,
              inverse_problem=False, inverse_problem_guess=None,
              data_root='.', data_seed=0, data_aux_info=None,
              num_domain=1000, num_boundary=1000, num_initial=5000, include_ic=True,
              test_max_pts=400000):
    """
    Arguments
        - pde_name
        - pde_const - constant for PDE as an iterable (depend on which pde is used)
        - data_root - where the base directory for PDEBench data is kept
        - data_seed - seed for data (for PDEBench)
    
    Returns tuple of
        - DeepXDE Data object
        - Dictionary containing all other intermediate items that were generated
    """
    if data_aux_info is None:
        data_aux_info = dict()
    
    if use_pdebench:
    
        if pde_name == 'reacdiff-1d':
            nu, rho = pde_const
            data, ext_vars, dataset = setup_pde1D(
                root_path=data_root,
                filename=f'1D/ReactionDiffusion/Train/ReacDiff_Nu{nu}_Rho{rho}.hdf5',
                aux_params=pde_const,
                xL=0.,
                xR=1.,
                val_batch_idx=data_seed,
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
                inverse_problem=inverse_problem, 
                inverse_problem_guess=inverse_problem_guess,
            )
            aux = {'dataset': dataset}
            
        elif pde_name == 'conv-1d':
            beta = pde_const[0]
            data, ext_vars, dataset = setup_pde1D(
                root_path=data_root,
                filename=f'1D/Advection/Train/1D_Advection_Sols_beta{beta}.hdf5',
                aux_params=pde_const,
                xL=0.,
                xR=1.,
                val_batch_idx=data_seed,
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
                inverse_problem=inverse_problem, 
                inverse_problem_guess=inverse_problem_guess,
            )
            aux = {'dataset': dataset}
        
        elif pde_name == 'burgers-1d':
            nu = pde_const[0]
            data, ext_vars, dataset = setup_pde1D(
                root_path=data_root,
                filename=f'1D/Burgers/Train/1D_Burgers_Sols_Nu{nu}.hdf5',
                aux_params=pde_const,
                xL=-1.,
                xR=1.,
                val_batch_idx=data_seed,
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
                inverse_problem=inverse_problem, 
                inverse_problem_guess=inverse_problem_guess,
            )
            aux = {'dataset': dataset}
            
        elif pde_name == 'cfdp-1d':
            gamma = pde_const[0]
            gamma_str = '1.e-8' if gamma == 1e-8 else str(gamma) #0.1. 0.01. 1e-8
            data, ext_vars, dataset = setup_pde1D(
                root_path=data_root,
                filename=f'1D/CFD/Train/1D_CFD_Rand_Eta{gamma_str}_Zeta{gamma_str}_periodic_Train.hdf5',
                aux_params=pde_const,
                xL=-1.,
                xR=1.,
                val_batch_idx=data_seed,
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
                inverse_problem=inverse_problem, 
                inverse_problem_guess=inverse_problem_guess,
            )
            aux = {'dataset': dataset}
            
        elif pde_name == 'cfdt-1d':
            data, ext_vars, dataset = setup_pde1D(
                root_path=data_root,
                filename=f'1D/CFD/Train/1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5',
                aux_params=pde_const,
                xL=-1.,
                xR=1.,
                val_batch_idx=data_seed,
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
                inverse_problem=inverse_problem, 
                inverse_problem_guess=inverse_problem_guess,
            )
            aux = {'dataset': dataset}
            
        # elif pde_name == 'diffsorp-1d':
        #     model, dataset = setup_diffusion_sorption(
        #         filename=os.path.join(data_root, '1D/diffusion-sorption/1D_diff-sorp_NA_NA.h5'),
        #         seed=f'{data_seed:04d}'
        #     )
            
        elif pde_name == 'reacdiff-2d':
            
            # only one data so constant must fit exactly
            assert len(pde_const) == 2
            assert pde_const[0] == 1e-3
            assert pde_const[1] == 5e-3
            
            data, dataset = setup_diffusion_reaction(
                filename=os.path.join(data_root, '2D/diffusion-reaction/2D_diff-react_NA_NA.h5'),
                seed=f'{data_seed:04d}',
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
            )
            
            aux = {
                'dataset': dataset,
                # 'anc_candidate_filter': lambda xs: xs[xs[:, 2] == jnp.floor(xs[:, 2])]  # only select points at selected timesteps
            }
            
            if inverse_problem:
                ext_vars = [dde.Variable(v) for v in inverse_problem_guess]
            else:
                ext_vars = None
                
        elif pde_name == 'darcy-2d':
            beta = pde_const[0]
            data, ext_vars, dataset = setup_darcy(
                filename=os.path.join(data_root, f'2D/DarcyFlow/2D_DarcyFlow_beta{beta}_Train.hdf5'),
                aux_params=pde_const,
                val_batch_idx=data_seed,
                num_domain=num_domain,
                num_boundary=num_boundary,
                num_initial=num_initial,
                inverse_problem=inverse_problem, 
                inverse_problem_guess=inverse_problem_guess,
            )
            aux = dict()
            
        elif pde_name == 'sw-2d':
            data, dataset = setup_swe_2d(
                filename=os.path.join(data_root, '2D/shallow-water/2D_rdb_NA_NA.h5'),
                seed=f'{data_seed:04d}'
            )
            ext_vars = None
            aux = {
                'dataset': dataset,
            }
            
        # elif pde_name == 'cfdp-2d':
        #     data, dataset = setup_CFD2D(
        #         root_path=data_root,
        #         filename=f'2D/NS_incom/ns_incom_inhom_2d_512-0.h5',
        #         val_batch_idx=data_seed,
        #         num_domain=num_domain,
        #         num_boundary=num_boundary,
        #         num_initial=num_initial,
        #     )
        #     ext_param = None
        
        else:
            raise ValueError(f'For use_pdebench=True, invalid pde_name "{pde_name}".')
        
        if not include_ic:
            data.bcs = [bc for bc in data.bcs if not (isinstance(bc, dde.icbc.IC) or isinstance(bc, dde.icbc.PointSetBC))]
    
    else:
        
        if (not inverse_problem) or (inverse_problem and (inverse_problem_guess is None)):
            inverse_problem_guess = pde_const
        
        if pde_name == 'heat-1d':
                        
            # Problem parameters:
            L = 1  # Length of the bar
            n = 1  # Frequency of the sinusoidal initial conditions
            
            def heat_eq_exact_solution(x, t):
                return (jnp.exp(-(n**2 * jnp.pi**2 * pde_const[0] * t) / (L**2)) * jnp.sin(n * jnp.pi * x / L)).reshape(-1, 1)

            def func(x):
                return heat_eq_exact_solution(x[:,0], x[:,1])


            def _pde(x, y, const):
                a = const[0]  # thermal diffusivity
                """Expresses the PDE residual of the heat equation."""
                dy_t = dde.grad.jacobian(y, x, i=0, j=1)[0]
                dy_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)[0]
                return (dy_t - a * dy_xx,)


            # Computational geometry:
            geom = dde.geometry.Interval(0, L)
            timedomain = dde.geometry.TimeDomain(0, 1)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)

            # Initial and boundary conditions:
            bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
            ic = dde.icbc.IC(
                geomtime,
                lambda x: jnp.sin(n * jnp.pi * x[:, 0:1] / L),
                lambda _, on_initial: on_initial,
            )
            icbc = [ic, bc]
            
            aux = {
                'func': func
            }
            
        elif pde_name == 'diff-1d':

            def func(x):
                return jnp.sin(jnp.pi * x[:, 0:1]) * jnp.exp(-x[:, 1:])
            
            # def transform_fn(x, y):
            #     return x[..., 1:2] * (1 - x[..., 0:1] ** 2) * y[..., 0:1] + jnp.sin(jnp.pi * x[..., 0:1])


            def _pde(x, y, const):
                dy_t = dde.grad.jacobian(y, x, i=0, j=1)[0]
                dy_xx = dde.grad.hessian(y, x, i=0, j=0)[0]
                # Backend tensorflow.compat.v1 or tensorflow
                return (
                    dy_t
                    - dy_xx
                    + jnp.exp(-x[:, 1:])
                    * (jnp.sin(jnp.pi * x[:, 0:1]) - jnp.pi ** 2 * jnp.sin(jnp.pi * x[:, 0:1]))
                )


            # Computational geometry:
            geom = dde.geometry.Interval(-1, 1)
            timedomain = dde.geometry.TimeDomain(0, 1)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)

            # Initial and boundary conditions:
            bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
            ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
            icbc = [ic, bc]
            
            aux = {
                'func': func,
            }
            
        elif pde_name == 'diffhc-1d':
    
            # hard constraint version
            l1 = 2.
            l2 = 4.
    
            def func(x):
                return jnp.sin(l2 * jnp.pi * x[:, 0:1]) * jnp.exp(- l1 * x[:, 1:])

            def transform_fn(x, y):
                return x[..., 1:2] * (1 - x[..., 0:1] ** 2) * y[..., 0:1] + jnp.sin(l2 * jnp.pi * x[..., 0:1])

            def _pde(x, y, const):
                
                def diff_t(g_):
                    h_ = lambda x_: g_(x_.reshape(1, -1))[0, 0]
                    return lambda x_: jax.vmap(jax.grad(h_), in_axes=0)(x_)[:, 1:2]
            
                def diff_x(g_):
                    h_ = lambda x_: g_(x_.reshape(1, -1))[0, 0]
                    return lambda x_: jax.vmap(jax.grad(h_), in_axes=0)(x_)[:, 0:1]
                    
                y_fn = y[1]
                dy_t = diff_t(y_fn)(x)
                dy_xx = diff_x(diff_x(y_fn))(x)
                
                return (
                    dy_t - dy_xx + jnp.exp(- l1 * x[:, 1:]) * jnp.sin(l2 * jnp.pi * x[:, 0:1]) * (l1 - (jnp.pi * l2) ** 2),
                )


            # Computational geometry:
            geom = dde.geometry.Interval(-1, 1)
            timedomain = dde.geometry.TimeDomain(0, 3)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)

            # Initial and boundary conditions:
            icbc = []
            
            aux = {
                'func': func,
                'out_transform': transform_fn,
            }
            
        elif pde_name == 'conv-1d':
                        
            beta = dde.Variable(inverse_problem_guess[0])
            L = data_aux_info.get('L', 1)  # Length of the bar
            n = data_aux_info.get('n', 1)  # Frequency of the sinusoidal initial conditions
            t_max = data_aux_info.get('t_max', 2)
            
            def _pde(x, y, const):
                beta = const[0]
                dy_t = dde.grad.jacobian(y, x, j=1)[0]
                dy_x = dde.grad.jacobian(y, x, j=0)[0]
                return (dy_t + beta * dy_x,)
            
            f_init = data_aux_info.get('f_init', lambda x_: jnp.sin(2. * n * jnp.pi * x_ / L))

            def func(x):
                return f_init(x[:, 0:1] - pde_const[0] * x[:, 1:2])


            geom = dde.geometry.Interval(0, L)
            timedomain = dde.geometry.TimeDomain(0, t_max)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
            
            def _boundary_r(x, on_boundary, xL, xR):
                return (on_boundary and jnp.isclose(x[0], xL)) or (on_boundary and jnp.isclose(x[0], xR))
            
            boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, 0, L)

            bc = dde.icbc.boundary_conditions.PeriodicBC(geomtime, 0, boundary_r)
            ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
            icbc = [ic, bc]
            
            aux = {
                'L': L,
                't_max': t_max,
                'f_init': f_init,
                'func': func
            }
            
        elif pde_name == 'kdv-1d':
                        
            from scipy.io import loadmat
            
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            data = loadmat(os.path.join(curr_dir, 'dataset/KdV.mat'))
            t_star = data['tt'].flatten()[:,None]
            x_star = data['x'].flatten()[:,None]
            Exact = np.real(data['uu'])
            
            xs = np.array(*[np.meshgrid(x_star.flatten(), t_star.flatten())]).reshape(2, -1).T
            ys = Exact.T.reshape(-1, 1)
            
            assert pde_const[0] == 1.  # 1.0
            assert pde_const[1] == 0.  # 0.0025
            
            def _pde(x, y, const):
                
                # implementing the whole thing without dde.grad.jacobian
                # since deepxde + jax + higher-order diff does not work well
                
                y_val = y[0]
                y_fn = y[1]
                # lambda_1, lambda_2 = const
                lambda_1 = const[0]
                lambda_2 = 0.0025 * jnp.exp(const[1])
                
                def diff_t(g_):
                    h_ = lambda x_: g_(x_.reshape(1, -1))[0, 0]
                    return lambda x_: jax.vmap(jax.grad(h_), in_axes=0)(x_)[:, 1:2]
                
                def diff_x(g_):
                    h_ = lambda x_: g_(x_.reshape(1, -1))[0, 0]
                    return lambda x_: jax.vmap(jax.grad(h_), in_axes=0)(x_)[:, 0:1]
                
                dy_t = diff_t(y_fn)(x)
                dy_x = diff_x(y_fn)(x)
                dy_xxx = diff_x(diff_x(diff_x(y_fn)))(x)
                    
                # def dy_x_fn(x_):
                #     y_ = y_fn(x_)[..., 0]
                #     print(x_.shape, y_.shape)
                #     return dde.grad.jacobian((y_.reshape(-1, 1), y_fn), x_.reshape(-1, 2), j=0)[0]

                # dy_t = dde.grad.jacobian(y, x, j=1)[0]
                # dy_x = dde.grad.jacobian(y, x, j=0)[0]
                # print(x.shape, dy_x.shape)
                # dy_xxx = dde.grad.hessian((dy_x, dy_x_fn), x, i=0, j=0)[0]
                
                return (dy_t + lambda_1 * y_val * dy_x + lambda_2 * dy_xxx,)


            def init_func(x):
                return jnp.cos(jnp.pi * x[:, 0:1])

            geom = dde.geometry.Interval(-1., 1.)
            timedomain = dde.geometry.TimeDomain(0, 1.)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
            
            def _boundary_r(x, on_boundary, xL, xR):
                return (on_boundary and jnp.isclose(x[0], xL)) or (on_boundary and jnp.isclose(x[0], xR))
            
            boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, -1, 1)

            bc = dde.icbc.boundary_conditions.PeriodicBC(geomtime, 0, boundary_r)
            ic = dde.icbc.IC(geomtime, init_func, lambda _, on_initial: on_initial)
            icbc = [ic, bc]
            
            func = None
            aux = {
                'test_x': jnp.array(xs),
                'test_y': jnp.array(ys),
            }
            
        elif pde_name == 'reacdiff-1d':
            
            # adapted from https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/reaction.inverse.html
            
            assert pde_const[0] == 0.1
            assert pde_const[1] == 2.
            
            def gen_traindata():
                curr_dir = os.path.dirname(os.path.realpath(__file__))
                data = np.load(os.path.join(curr_dir, "dataset/reaction.npz"))
                t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
                X, T = np.meshgrid(x, t)
                X = np.reshape(X, (-1, 1))
                T = np.reshape(T, (-1, 1))
                Ca = np.reshape(ca, (-1, 1))
                Cb = np.reshape(cb, (-1, 1))
                return np.hstack((X, T)), np.hstack((Ca, Cb))


            def _pde(x, y, const):
                kf, D = const
                ca, cb = y[0][:, 0:1], y[0][:, 1:2]
                dca_t = dde.grad.jacobian(y, x, i=0, j=1)[0]
                dca_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)[0]
                dcb_t = dde.grad.jacobian(y, x, i=1, j=1)[0]
                dcb_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)[0]
                eq_a = dca_t - 1e-3 * D * dca_xx + kf * ca * cb ** 2
                eq_b = dcb_t - 1e-3 * D * dcb_xx + 2 * kf * ca * cb ** 2
                return (jnp.sqrt(eq_a**2 + eq_b**2),)


            def fun_bc(x):
                return 1 - x[:, 0:1]


            def fun_init(x):
                return jnp.exp(-20 * x[:, 0:1])


            geom = dde.geometry.Interval(0, 1)
            timedomain = dde.geometry.TimeDomain(0, 10)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)

            bc_a = dde.icbc.DirichletBC(
                geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=0
            )
            bc_b = dde.icbc.DirichletBC(
                geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
            )
            ic1 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
            ic2 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)

            observe_x, observe_y = gen_traindata()

            icbc = [bc_a, bc_b, ic1, ic2]
            
            func = None
            aux = {
                'test_x': observe_x,
                'test_y': observe_y,
            }
            
        elif pde_name == 'fd-2d':
            
            from scipy.io import loadmat
            
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            
            assert pde_const[0] == 1.0
            assert pde_const[1] == 0.01
            # C1true = 1.0
            # C2true = 0.01
            
            # Load training data
            def load_training_data():
                data = loadmat(os.path.join(curr_dir, "dataset/cylinder_nektar_wake.mat"))
                U_star = data["U_star"]  # N x 2 x T
                P_star = data["p_star"]  # N x T
                t_star = data["t"]  # T x 1
                X_star = data["X_star"]  # N x 2
                N = X_star.shape[0]
                T = t_star.shape[0]
                # Rearrange Data
                XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
                YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
                TT = np.tile(t_star, (1, N)).T  # N x T
                UU = U_star[:, 0, :]  # N x T
                VV = U_star[:, 1, :]  # N x T
                PP = P_star  # N x T
                x = XX.flatten()[:, None]  # NT x 1
                y = YY.flatten()[:, None]  # NT x 1
                t = TT.flatten()[:, None]  # NT x 1
                u = UU.flatten()[:, None]  # NT x 1
                v = VV.flatten()[:, None]  # NT x 1
                p = PP.flatten()[:, None]  # NT x 1
                # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
                data1 = np.concatenate([x, y, t, u, v, p], 1)
                data2 = data1[:, :][data1[:, 2] <= 7]
                data3 = data2[:, :][data2[:, 0] >= 1]
                data4 = data3[:, :][data3[:, 0] <= 8]
                data5 = data4[:, :][data4[:, 1] >= -2]
                data_domain = data5[:, :][data5[:, 1] <= 2]
                # choose number of training points: num =7000
                # idx = np.random.choice(data_domain.shape[0], num, replace=False)
                # x_train = data_domain[:, 0:1]
                # y_train = data_domain[:, 1:2]
                # t_train = data_domain[:, 2:3]
                # u_train = data_domain[:, 3:4]
                # v_train = data_domain[:, 4:5]
                # p_train = data_domain[:, 5:6]
                # return [x_train, y_train, t_train, u_train, v_train, p_train]
                
                return data_domain[:, 0:3], data_domain[:, 3:6]

            # Define Navier Stokes Equations (Time-dependent PDEs)
            def _pde(x, y, const):
                
                C1, C2 = const
                
                u = y[0][:, 0:1]
                v = y[0][:, 1:2]
                p = y[0][:, 2:3]
                
                du_x = dde.grad.jacobian(y, x, i=0, j=0)[0]
                du_y = dde.grad.jacobian(y, x, i=0, j=1)[0]
                du_t = dde.grad.jacobian(y, x, i=0, j=2)[0]
                dv_x = dde.grad.jacobian(y, x, i=1, j=0)[0]
                dv_y = dde.grad.jacobian(y, x, i=1, j=1)[0]
                dv_t = dde.grad.jacobian(y, x, i=1, j=2)[0]
                dp_x = dde.grad.jacobian(y, x, i=2, j=0)[0]
                dp_y = dde.grad.jacobian(y, x, i=2, j=1)[0]
                du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)[0]
                du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)[0]
                dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)[0]
                dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)[0]
                
                continuity = du_x + dv_y
                x_momentum = du_t + C1 * (u * du_x + v * du_y) + dp_x - C2 * (du_xx + du_yy)
                y_momentum = dv_t + C1 * (u * dv_x + v * dv_y) + dp_y - C2 * (dv_xx + dv_yy)
                
                return (jnp.sqrt(continuity**2 + x_momentum**2 + y_momentum**2),)
                # return [continuity, x_momentum, y_momentum]

            # Define Spatio-temporal domain
            # Rectangular
            Lx_min, Lx_max = 1.0, 8.0
            Ly_min, Ly_max = -2.0, 2.0
            # Spatial domain: X × Y = [1, 8] × [−2, 2]
            space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
            # Time domain: T = [0, 7]
            time_domain = dde.geometry.TimeDomain(0, 7)
            # Spatio-temporal domain
            geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

            # Get the training data: num = 7000
            ob_xyt_all, ob_uvp_all = load_training_data()
            
            idxs = jnp.array(np.random.default_rng(seed=42).choice(ob_xyt_all.shape[0], size=20000, replace=False))
            ob_xyt = ob_xyt_all[idxs]
            ob_uvp = ob_uvp_all[idxs]
            observe_u = dde.icbc.PointSetBC(ob_xyt, ob_uvp[:, 0:1], component=0)
            observe_v = dde.icbc.PointSetBC(ob_xyt, ob_uvp[:, 1:2], component=1)

            # # Training datasets and Loss
            # data = dde.data.TimePDE(
            #     geomtime,
            #     Navier_Stokes_Equation,
            #     [observe_u, observe_v],
            #     num_domain=num_domain,
            #     num_boundary=num_boundary,
            #     num_initial=num_initial,
            # )
            
            icbc = [observe_u, observe_v]
            func = None
            aux = {
                'test_x': ob_xyt,
                'test_y': ob_uvp,
            }
            
        elif pde_name == 'eik1-2d':
            time_dep = False
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            
            filename_T = "dataset/eikonal_2d.npy"
            xy, zs = np.load(os.path.join(curr_dir, filename_T))
            sx = 0.
            sy = 0.
            
            norm_scale = np.linalg.norm(xy - np.array([sx, sy]), axis=1)
            zs[:,0] = zs[:,0] / norm_scale
            # print(np.argwhere(np.isnan(y)))
            # solving for singularity
            zs[0,0] = (zs[1,0] + zs[101,0] + zs[102,0]) / 3
            
            def T_transform(x_, y_):
                dx1 = x_[..., 0:1] - sx
                dx2 = x_[..., 1:2] - sy
                dist = jnp.maximum(dx1**2 + dx2**2, 1e-12)
                source_dist = jnp.sqrt(dist)
                return y_[..., 0:1] * source_dist
            
            def _pde(x, y, const):
                
                # NN output = [T, marm]
                y_fn = y[1]
                
                T = lambda x_: T_transform(x_, y_fn(x_))
                
                dT_dx = dde.grad.jacobian((T(x), T), x, i=0, j=0)[0]
                dT_dy = dde.grad.jacobian((T(x), T), x, i=0, j=1)[0]
                                
                T_mag = jnp.sqrt(dT_dx**2 + dT_dy**2)
                
                v_mag = y[0][:, 1:2]
                
                return (T_mag * v_mag - 1.,)
            
            geomtime = dde.geometry.Rectangle(xmin=[0., 0.], xmax=[4., 4.])
            icbc = [
                dde.icbc.PointSetBC(jnp.array(xy), jnp.array(zs[:,1:2]), component=1)
            ]
                
            func = None  
            aux = {
                'test_x': jnp.array(xy),
                'test_y': jnp.array(zs),
                'x_test_norm_scale': norm_scale,
                'T_transform': T_transform,
            }
            
        elif pde_name == 'eik3-2d':
            time_dep = False
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            
            filename_T = "dataset/eikonal_2d-1.npy"
            xy, zs = np.load(os.path.join(curr_dir, filename_T))
            sx = 0.
            sy = 0.
            
            norm_scale = np.linalg.norm(xy - np.array([sx, sy]), axis=1)
            zs[:,0] = zs[:,0] / norm_scale
            # print(np.argwhere(np.isnan(y)))
            # solving for singularity
            zs[0,0] = (zs[1,0] + zs[101,0] + zs[102,0]) / 3
            
            def T_transform(x_, y_):
                dx1 = x_[..., 0:1] - sx
                dx2 = x_[..., 1:2] - sy
                dist = jnp.maximum(dx1**2 + dx2**2, 1e-12)
                source_dist = jnp.sqrt(dist)
                return y_[..., 0:1] * source_dist
            
            def _pde(x, y, const):
                
                # NN output = [T, marm]
                y_fn = y[1]
                
                T = lambda x_: T_transform(x_, y_fn(x_))
                
                dT_dx = dde.grad.jacobian((T(x), T), x, i=0, j=0)[0]
                dT_dy = dde.grad.jacobian((T(x), T), x, i=0, j=1)[0]
                                
                T_mag = jnp.sqrt(dT_dx**2 + dT_dy**2)
                
                v_mag = y[0][:, 1:2]
                
                return (T_mag * v_mag - 1.,)
            
            geomtime = dde.geometry.Rectangle(xmin=[0., 0.], xmax=[4., 4.])
            icbc = [
                dde.icbc.PointSetBC(jnp.array(xy), jnp.array(zs[:,1:2]), component=1)
            ]
                
            func = None  
            aux = {
                'test_x': jnp.array(xy),
                'test_y': jnp.array(zs),
                'x_test_norm_scale': norm_scale,
                'T_transform': T_transform,
            }
            
            
        elif pde_name == 'eik0-3d':
            time_dep = False
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            
            filename_T = "dataset/eikonal_3d.npy"
            d = np.load(os.path.join(curr_dir, filename_T))
            xy = d[:,0:3]
            zs = d[:,3:5]
            
            def _pde(x, y, const):
                
                # NN output = [T, marm]
                y_fn = y[1]
                
                dT_dx = dde.grad.jacobian(y, x, i=0, j=0)[0]
                dT_dy = dde.grad.jacobian(y, x, i=0, j=1)[0]
                dT_dz = dde.grad.jacobian(y, x, i=0, j=2)[0]
                T_mag = jnp.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2)
                
                v_mag = y[0][:, 1:2]
                
                return (T_mag * v_mag - 1.,)
            
            geomtime = dde.geometry.Cuboid(xmin=[0., 0., 0.], xmax=[2., 2., 2.])
            icbc = [
                dde.icbc.PointSetBC(jnp.array(xy), jnp.array(zs[:,1]).reshape(-1, 1), component=1)
            ]
                
            func = None  
            aux = {
                'test_x': jnp.array(xy),
                'test_y': jnp.array(zs),
            }
            
        elif pde_name == 'eik1-3d':
            time_dep = False
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            
            filename_T = "dataset/eikonal_3d.npy"
            d = np.load(os.path.join(curr_dir, filename_T))
            xy = d[:,0:3]
            zs = d[:,3:5]
            
            sx = 0.
            sy = 0.
            sz = 0.
            
            norm_scale = np.linalg.norm(xy - np.array([sx, sy, sz]), axis=1)
            zs[:,0] = zs[:,0] / norm_scale
            # print(np.argwhere(np.isnan(y)))
            # solving for singularity
            zs[0,0] = zs[1,0]
            
            def T_transform(x_, y_):
                dx1 = x_[..., 0:1] - sx
                dx2 = x_[..., 1:2] - sy
                dx3 = x_[..., 2:3] - sy
                dist = jnp.maximum(dx1**2 + dx2**2 + dx3**2, 1e-12)
                source_dist = jnp.sqrt(dist)
                return y_[..., 0:1] * source_dist
            
            def _pde(x, y, const):
                
                # NN output = [T, marm]
                y_fn = y[1]
                
                T = lambda x_: T_transform(x_, y_fn(x_))
                
                dT_dx = dde.grad.jacobian((T(x), T), x, i=0, j=0)[0]
                dT_dy = dde.grad.jacobian((T(x), T), x, i=0, j=1)[0]
                dT_dz = dde.grad.jacobian((T(x), T), x, i=0, j=2)[0]            
                T_mag = jnp.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2)
                
                v_mag = y[0][:, 1:2]
                
                return (T_mag * v_mag - 1.,)
            
            geomtime = dde.geometry.Cuboid(xmin=[0., 0., 0.], xmax=[2., 2., 2.])
            icbc = [
                dde.icbc.PointSetBC(jnp.array(xy), jnp.array(zs[:,1:2]), component=1)
            ]
                
            func = None  
            aux = {
                'test_x': jnp.array(xy),
                'test_y': jnp.array(zs),
                'x_test_norm_scale': norm_scale,
                'T_transform': T_transform,
            }
            
        elif pde_name == 'kf-2d':
            
            time_dep = False

            def _pde(x, u, const):
                
                Re = const[0]
                
                u_vel, v_vel, p = u[0][:, 0:1], u[0][:, 1:2], u[0][:, 2:]
                
                u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)[0]
                u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)[0]
                u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)[0]
                u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)[0]

                v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)[0]
                v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)[0]
                v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)[0]
                v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)[0]

                p_x = dde.grad.jacobian(u, x, i=2, j=0)[0]
                p_y = dde.grad.jacobian(u, x, i=2, j=1)[0]

                momentum_x = (
                    u_vel * u_vel_x + v_vel * u_vel_y + p_x - (1 / Re) * (u_vel_xx + u_vel_yy)
                )
                momentum_y = (
                    u_vel * v_vel_x + v_vel * v_vel_y + p_y - (1 / Re) * (v_vel_xx + v_vel_yy)
                )
                continuity = u_vel_x + v_vel_y

                # return [momentum_x, momentum_y, continuity]
                return (jnp.sqrt(continuity**2 + momentum_x**2 + momentum_y**2),)


            Re = pde_const[0]  # 20
            nu = 1 / Re
            l = 1 / (2 * nu) - jnp.sqrt(1 / (4 * nu**2) + 4 * jnp.pi**2)

            def u_func(x):
                return 1 - jnp.exp(l * x[:, 0:1]) * jnp.cos(2 * jnp.pi * x[:, 1:2])

            def v_func(x):
                return l / (2 * jnp.pi) * jnp.exp(l * x[:, 0:1]) * jnp.sin(2 * jnp.pi * x[:, 1:2])

            def p_func(x):
                return 1 / 2 * (1 - jnp.exp(2 * l * x[:, 0:1]))
            
            def func(x):
                return jnp.concatenate([u_func(x), v_func(x), p_func(x)], axis=1)
            

            def boundary_outflow(x, on_boundary):
                return on_boundary and jnp.isclose(x[0], 1)


            spatial_domain = dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

            boundary_condition_u = dde.icbc.DirichletBC(
                spatial_domain, u_func, lambda _, on_boundary: on_boundary, component=0
            )
            boundary_condition_v = dde.icbc.DirichletBC(
                spatial_domain, v_func, lambda _, on_boundary: on_boundary, component=1
            )
            # boundary_condition_right_p = dde.icbc.DirichletBC(
            #     spatial_domain, p_func, boundary_outflow, component=2
            # )
            
            right_p = jnp.stack([1. * jnp.ones(shape=(200,)), jnp.linspace(-0.5, 1.5, num=200)], axis=1)
            boundary_condition_right_p = dde.icbc.PointSetBC(right_p, p_func(right_p), component=2)

            # data = dde.data.PDE(
            #     spatial_domain,
            #     pde,
            #     [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
            #     num_domain=2601,
            #     num_boundary=400,
            #     num_test=100000,
            # )  
            
            geomtime = spatial_domain
            # pde = pde
            icbc = [boundary_condition_u, boundary_condition_v, boundary_condition_right_p]
            
            aux = {
                'func': func,
                'Re': Re
            }
            
        elif pde_name == 'beltrami-3d':
            # https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Beltrami_flow.py
            
            time_dep = True
            
            a = 1
            d = 1
            Re = 1

            def _pde(x, u, const):
                
                u_vel, v_vel, w_vel, p = u[0][:, 0:1], u[0][:, 1:2], u[0][:, 2:3], u[0][:, 3:4]

                u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)[0]
                u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)[0]
                u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)[0]
                u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)[0]
                u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)[0]
                u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)[0]
                u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)[0]

                v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)[0]
                v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)[0]
                v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)[0]
                v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)[0]
                v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)[0]
                v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)[0]
                v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)[0]

                w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)[0]
                w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)[0]
                w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)[0]
                w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)[0]
                w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)[0]
                w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)[0]
                w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)[0]

                p_x = dde.grad.jacobian(u, x, i=3, j=0)[0]
                p_y = dde.grad.jacobian(u, x, i=3, j=1)[0]
                p_z = dde.grad.jacobian(u, x, i=3, j=2)[0]

                momentum_x = (
                    u_vel_t
                    + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
                    + p_x
                    - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
                )
                momentum_y = (
                    v_vel_t
                    + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
                    + p_y
                    - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
                )
                momentum_z = (
                    w_vel_t
                    + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
                    + p_z
                    - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
                )
                continuity = u_vel_x + v_vel_y + w_vel_z

                # return [momentum_x, momentum_y, momentum_z, continuity]
                return (jnp.sqrt(continuity**2 + momentum_x**2 + momentum_y**2 + momentum_z**2),)
            
            def u_func(x):
                return (
                    -a
                    * (
                        jnp.exp(a * x[:, 0:1]) * jnp.sin(a * x[:, 1:2] + d * x[:, 2:3])
                        + jnp.exp(a * x[:, 2:3]) * jnp.cos(a * x[:, 0:1] + d * x[:, 1:2])
                    )
                    * jnp.exp(-(d ** 2) * x[:, 3:4])
                )


            def v_func(x):
                return (
                    -a
                    * (
                        jnp.exp(a * x[:, 1:2]) * jnp.sin(a * x[:, 2:3] + d * x[:, 0:1])
                        + jnp.exp(a * x[:, 0:1]) * jnp.cos(a * x[:, 1:2] + d * x[:, 2:3])
                    )
                    * jnp.exp(-(d ** 2) * x[:, 3:4])
                )


            def w_func(x):
                return (
                    -a
                    * (
                        jnp.exp(a * x[:, 2:3]) * jnp.sin(a * x[:, 0:1] + d * x[:, 1:2])
                        + jnp.exp(a * x[:, 1:2]) * jnp.cos(a * x[:, 2:3] + d * x[:, 0:1])
                    )
                    * jnp.exp(-(d ** 2) * x[:, 3:4])
                )


            def p_func(x):
                return (
                    -0.5
                    * a ** 2
                    * (
                        jnp.exp(2 * a * x[:, 0:1])
                        + jnp.exp(2 * a * x[:, 1:2])
                        + jnp.exp(2 * a * x[:, 2:3])
                        + 2
                        * jnp.sin(a * x[:, 0:1] + d * x[:, 1:2])
                        * jnp.cos(a * x[:, 2:3] + d * x[:, 0:1])
                        * jnp.exp(a * (x[:, 1:2] + x[:, 2:3]))
                        + 2
                        * jnp.sin(a * x[:, 1:2] + d * x[:, 2:3])
                        * jnp.cos(a * x[:, 0:1] + d * x[:, 1:2])
                        * jnp.exp(a * (x[:, 2:3] + x[:, 0:1]))
                        + 2
                        * jnp.sin(a * x[:, 2:3] + d * x[:, 0:1])
                        * jnp.cos(a * x[:, 1:2] + d * x[:, 2:3])
                        * jnp.exp(a * (x[:, 0:1] + x[:, 1:2]))
                    )
                    * jnp.exp(-2 * d ** 2 * x[:, 3:4])
                )
                
            spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
            temporal_domain = dde.geometry.TimeDomain(0, 1)
            geomtime = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

            boundary_condition_u = dde.icbc.DirichletBC(
                geomtime, u_func, lambda _, on_boundary: on_boundary, component=0
            )
            boundary_condition_v = dde.icbc.DirichletBC(
                geomtime, v_func, lambda _, on_boundary: on_boundary, component=1
            )
            boundary_condition_w = dde.icbc.DirichletBC(
                geomtime, w_func, lambda _, on_boundary: on_boundary, component=2
            )

            initial_condition_u = dde.icbc.IC(
                geomtime, u_func, lambda _, on_initial: on_initial, component=0
            )
            initial_condition_v = dde.icbc.IC(
                geomtime, v_func, lambda _, on_initial: on_initial, component=1
            )
            initial_condition_w = dde.icbc.IC(
                geomtime, w_func, lambda _, on_initial: on_initial, component=2
            )
            
            icbc = [
                boundary_condition_u,
                boundary_condition_v,
                boundary_condition_w,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
            ]
            
            x, y, z, t = np.meshgrid(
                np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.linspace(1, 1, 50)
            )
            
            test_x = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
            test_y = np.concatenate((u_func(test_x), v_func(test_x), w_func(test_x), p_func(test_x)), axis=1)
            
            assert test_x.shape == test_y.shape == (50**4, 4), (test_x.shape, test_y.shape)
            
            func = None
            aux = {
                'test_x': jnp.array(test_x),
                'test_y': jnp.array(test_y),
                'u_func': u_func,
                'v_func': v_func,
                'w_func': w_func,
                'p_func': p_func,
            }
        
        else:
            raise ValueError(f'For use_pdebench=False, invalid pde_name "{pde_name}".')
        
        if inverse_problem:
            pde = _pde
            ext_vars = [dde.Variable(v) for v in inverse_problem_guess]
        else:
            pde = lambda x, y: _pde(x, y, const=pde_const)
        
        
        # in the case that we need to generate test case ourselves
        if func is not None:
        
            if isinstance(geomtime, dde.geometry.GeometryXTime):
                data = dde.data.TimePDE(
                    geomtime,
                    pde,
                    icbc,
                    num_domain=num_domain,
                    num_boundary=num_boundary,
                    num_initial=num_initial,
                    solution=func,
                    num_test=50000,
                )
                geom = geomtime.geometry
                if isinstance(geom, dde.geometry.Rectangle):
                    grid = jnp.meshgrid(jnp.linspace(geom.xmin[0], geom.xmax[0], 200), jnp.linspace(geom.xmin[1], geom.xmax[1], 200))
                else:
                    grid = jnp.meshgrid(jnp.linspace(geom.l, geom.r, 200), jnp.linspace(timedomain.t0, timedomain.t1, 200))
                
            elif isinstance(geomtime, dde.geometry.Rectangle):
                data = dde.data.PDE(
                    geomtime,
                    pde,
                    icbc,
                    num_domain=num_domain,
                    num_boundary=num_boundary,
                    solution=func,
                    num_test=50000,
                )
                grid = jnp.meshgrid(jnp.linspace(geomtime.xmin[0], geomtime.xmax[0], 200), jnp.linspace(geomtime.xmin[1], geomtime.xmax[1], 200))
            
            else:
                data = dde.data.PDE(
                    geomtime,
                    pde,
                    icbc,
                    num_domain=num_domain,
                    num_boundary=num_boundary,
                    solution=func,
                    num_test=1000,
                )
        
            data.test_x = jnp.array(grid).reshape(2, -1).T
            data.test_y = func(data.test_x)
            
        else:

            if isinstance(geomtime, dde.geometry.GeometryXTime):
                data = dde.data.TimePDE(
                    geomtime,
                    pde,
                    icbc,
                    num_domain=num_domain,
                    num_boundary=num_boundary,
                    num_initial=num_initial,
                    num_test=50000,
                )
                
            elif isinstance(geomtime, dde.geometry.Rectangle):
                data = dde.data.PDE(
                    geomtime,
                    pde,
                    icbc,
                    num_domain=num_domain,
                    num_boundary=num_boundary,
                    num_test=50000,
                )
            
            else:
                data = dde.data.PDE(
                    geomtime,
                    pde,
                    icbc,
                    num_domain=num_domain,
                    num_boundary=num_boundary,
                    num_test=1000,
                )
            
            data.test_x = jnp.array(aux['test_x'])
            data.test_y = jnp.array(aux['test_y'])
        
        if not include_ic:
            data.bcs = [bc for bc in data.bcs if not (isinstance(bc, dde.icbc.IC) or isinstance(bc, dde.icbc.PointSetBC) or isinstance(bc, dde.icbc.DirichletBC))]
        
    data.train_x = jnp.array(data.train_x)
    data.train_x_all = jnp.array(data.train_x_all)
    data.train_x_bc = jnp.array(data.train_x_bc)
    data.test_x = jnp.array(data.test_x)
    
    if data.test_x.shape[0] > test_max_pts:
        dim = data.test_x.shape[1]
        range_dim = [(jnp.min(data.test_x[:,i]), jnp.max(data.test_x[:,i])) for i in range(dim)]
        idxs_bc = jnp.where(sum([data.test_x[:,i] == m for i in range(dim) for m in range_dim[i]]) >= dim - 1)[0]
        # idxs = jnp.linspace(0, data.test_x.shape[0], num=test_max_pts, dtype=int)
        idxs = jnp.array(np.random.default_rng(seed=42).choice(data.test_x.shape[0], size=test_max_pts, replace=False))
        idxs = jnp.unique(jnp.concatenate([idxs, idxs_bc]))
        data.test_x = data.test_x[idxs]
        data.test_y = data.test_y[idxs]
    
    if not inverse_problem:
        ext_vars = None
    
    return data, ext_vars, aux
