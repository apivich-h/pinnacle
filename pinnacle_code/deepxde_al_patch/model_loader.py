from . import deepxde as dde

from .train_set_loader import load_data
from .models import generate_fourier_fnn, FNNWithLAAF, FNNWithGAAF


def construct_net(input_dim=2, hidden_layers=2, hidden_dim=64, output_dim=1, 
                  activation='tanh', initializer='Glorot uniform', 
                  arch=None, fourier_count=100, fourier_scale=1.):
    
    layer_size = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
    
    if arch == 'fourier':
        net, ffs = generate_fourier_fnn(
            layer_sizes=layer_size, activation=activation, kernel_initializer=initializer, 
            ff_count=fourier_count, W_scale=fourier_scale
        )
        aux = {'ffs': ffs}
        
    elif arch == 'laaf':
        net = FNNWithLAAF(layer_sizes=layer_size, activation=activation, kernel_initializer=initializer)
        aux = dict()

    elif arch == 'gaaf':
        print("GAFF mode")
        net = FNNWithGAAF(layer_sizes=layer_size, activation=activation, kernel_initializer=initializer)
        aux = dict()
        
    elif arch == 'pfnn':
        assert output_dim > 1
        layer_size_alt = [input_dim] + [[hidden_dim for _ in range(output_dim)]] * hidden_layers + [output_dim]
        net = dde.nn.PFNN(layer_sizes=layer_size_alt, activation=activation, kernel_initializer=initializer)
        aux = dict()
    
    else:
        net = dde.nn.FNN(layer_sizes=layer_size, activation=activation, kernel_initializer=initializer)
        aux = dict()
        
    return net, aux


def construct_model(pde_name, pde_const, use_pdebench=False, inverse_problem=False, inverse_problem_guess=None,
                    data_root='.', data_seed=0, data_aux_info=None, test_max_pts=400000,
                    num_domain=1000, num_boundary=1000, num_initial=5000, include_ic=True,
                    hidden_layers=2, hidden_dim=64, activation='tanh', initializer='Glorot uniform', 
                    arch=None, fourier_count=100, fourier_scale=1., 
                    do_compile=True, compile_optim='adam', compile_lr=1e-3):
    
    data, ext_vars, data_aux = load_data(
        pde_name=pde_name, pde_const=pde_const, use_pdebench=use_pdebench, test_max_pts=test_max_pts,
        inverse_problem=inverse_problem, inverse_problem_guess=inverse_problem_guess,
        data_root=data_root, data_seed=data_seed, data_aux_info=data_aux_info,
        num_domain=num_domain, num_boundary=num_boundary, num_initial=num_initial, include_ic=include_ic,
    )
    
    input_dim = data.test_x.shape[1]
    output_dim = data.test_y.shape[1]
    
    net, net_aux = construct_net(
        input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, output_dim=output_dim, 
        activation=activation, initializer=initializer, 
        arch=arch, fourier_count=fourier_count, fourier_scale=fourier_scale
    )
    
    model = dde.Model(data, net)
    if do_compile:
        model.compile(
            optimizer=compile_optim, lr=compile_lr, 
            metrics=["l2 relative error"], external_trainable_variables=ext_vars
        )
        
    aux = {
        'ext_params': ext_vars, 
        'data_aux': data_aux,
        'net_aux': net_aux
    }
    
    return model, aux
