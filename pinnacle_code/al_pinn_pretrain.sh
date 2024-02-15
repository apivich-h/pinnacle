#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2; al_pinn_pretrain.sh [pde code]

current_dir=$( dirname -- "$0"; )
echo $current_dir

pdes=(
    "--hidden_layers 4 --hidden_dim 128 --eqn conv-1d --init_const 40.0"
    "--hidden_layers 8 --hidden_dim 128 --eqn conv-1d --init_const 1.0 --use_pdebench --data_seed 40"
    "--hidden_layers 8 --hidden_dim 128 --eqn conv-1d --init_const 1.0 --use_pdebench --data_seed 80"
    "--hidden_layers 8 --hidden_dim 128 --eqn conv-1d --init_const 1.0 --use_pdebench --data_seed 898"
    "--hidden_layers 8 --hidden_dim 128 --eqn conv-1d --init_const 1.0 --use_pdebench --data_seed 272"
    "--hidden_layers 4 --hidden_dim 128 --eqn burgers-1d --use_pdebench --init_const 0.1 --data_seed 20"
    "--hidden_layers 4 --hidden_dim 128 --eqn burgers-1d --use_pdebench --init_const 0.1 --data_seed 40"
    "--hidden_layers 4 --hidden_dim 128 --eqn burgers-1d --use_pdebench --init_const 0.02 --data_seed 20"
    "--hidden_layers 4 --hidden_dim 128 --eqn burgers-1d --use_pdebench --init_const 0.02 --data_seed 131"
    "--hidden_layers 4 --hidden_dim 128 --eqn burgers-1d --use_pdebench --init_const 0.02 --data_seed 10"
)

losses=(
    "--loss_w_bcs 1.0"
)

for j in $3; do

    for k in $1; do

        pde="${pdes[$k]}"
        echo "PDE params: $pde"

        for loss in "${losses[@]}"; do

            pdeargs="$pde $loss --param_ver $j $2"

            echo "Run experiment with arg: ${pdeargs}"

            python ${current_dir}/al_pinn_pretrain.py $pdeargs

        done

    done

done