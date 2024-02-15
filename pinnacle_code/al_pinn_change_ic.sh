#!/bin/bash

# COMMAND
# al_pinn_change_ic.sh [pde code] [method] [param_ver list] [other params]

# EXAMPLE COMMAND
# al_pinn_change_ic.sh "0" "0 3 6" "0 1 2 3 4" "--pdebench_dir /home/a/apivich/pdebench"

current_dir=$( dirname -- "$0"; )
echo $current_dir

pdes=(
    "--hidden_layers 8 --hidden_dim 128 --eqn conv-1d --const 1.0 --data_seed_pair 898 40 --train_steps 200000 --al_every 5000 --num_points 50 --mem_pts_total_budget 200"
    "--hidden_layers 8 --hidden_dim 128 --eqn conv-1d --const 1.0 --data_seed_pair 272 80 --train_steps 200000 --al_every 5000 --num_points 50 --mem_pts_total_budget 200"
    "--hidden_layers 4 --hidden_dim 128 --eqn burgers-1d --const 0.02 --data_seed_pair 131 20 --train_steps 200000 --al_every 5000 --num_points 50 --mem_pts_total_budget 200"
)

algs=(
    "--method random --rand_method pseudo --rand_res_prop 0.8"
    "--method random --rand_method pseudo --rand_res_prop 0.5"
    "--method random --rand_method Hammersley --rand_res_prop 0.8"
    "--method random --rand_method Hammersley --rand_res_prop 0.5"
    "--method residue --res_res_prop 0.8"
    "--method residue --res_res_prop 0.8 --res_all_types"
    "--method residue --res_res_prop 0.5"
    "--method residue --res_res_prop 0.5 --res_all_types"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method kmeans --eig_weight_method alignment_norm --eig_sampling pseudo --eig_memory --auto_al"
    "--method sampling --eig_weight_method alignment_norm --eig_sampling pseudo --eig_memory --auto_al"
)

losses=(
    "--loss_w_bcs 1.0"
)

for j in $3; do

    for k in $1; do

        pde="${pdes[$k]}"
        echo "PDE params: $pde"

        for loss in "${losses[@]}"; do

            for m in $2; do

                alg="${algs[$m]}"

                pdeargs="$pde $alg $loss --param_ver $j $4"

                echo "Run experiment with arg: ${pdeargs}"

                python ${current_dir}/al_pinn_change_ic.py $pdeargs
                
            done

        done

    done

done