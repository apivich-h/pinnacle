#!/bin/bash

current_dir=$( dirname -- "$0"; )
echo $current_dir

# COMMAND
# al_pinn.sh {tests to run} {methods to run} {repeats} {other args for python script}

# EXAMPLE COMMAND
# al_pinn.sh "0" "0 3 6" "0 1 2 3 4" "--pdebench_dir /home/a/apivich/pdebench"

pdes=(
    "--hidden_layers 8 --eqn conv-1d --use_pdebench --data_seed 40 --const 1.0 --train_steps 200000 --num_points 200 --mem_pts_total_budget 1000"
    "--hidden_layers 8 --eqn conv-1d --use_pdebench --data_seed 80 --const 1.0 --train_steps 200000 --num_points 200 --mem_pts_total_budget 1000"
    "--hidden_layers 4 --eqn burgers-1d --use_pdebench --data_seed 20 --const 0.02 --train_steps 200000 --num_points 100 --mem_pts_total_budget 300"
    "--hidden_layers 4 --eqn burgers-1d --use_pdebench --data_seed 40 --const 0.02 --train_steps 200000 --num_points 100 --mem_pts_total_budget 300"
    "--hidden_layers 6 --nn laaf --hidden_dim 64 --eqn fd-2d --const 1.0 0.01 --no-allow_ic --anc_measurable_idx 0 1 --inverse --inverse_guess 0 0 --train_steps 100000 --al_every 1000 --select_anchors_every 1000 --num_points 200 --mem_pts_total_budget 1000 --anchor_budget 30"
    "--hidden_layers 8 --nn laaf --hidden_dim 32 --eqn eik1-3d --no-allow_ic --anc_measurable_idx 0 --train_steps 100000 --al_every 1000 --select_anchors_every 1000 --num_points 100 --mem_pts_total_budget 500 --anchor_budget 5"
    "--hidden_layers 8 --nn pfnn --hidden_dim 32 --eqn darcy-2d --use_pdebench --data_seed 20 --const 1.0 --no-allow_ic --anc_measurable_idx 1 --train_steps 100000 --al_every 1000 --num_points 200 --mem_pts_total_budget 1000 --select_anchors_every 1000 --anchor_budget 5"
    "--hidden_layers 4 --hidden_dim 32 --eqn sw-2d --use_pdebench --train_steps 100000 --al_every 5000 --num_points 200 --mem_pts_total_budget 1000"
    "--hidden_layers 4 --hidden_dim 32 --eqn kdv-1d --const 1.0 0.0 --train_steps 100000 --al_every 5000 --num_points 100 --mem_pts_total_budget 300"
    "--hidden_layers 4 --hidden_dim 32 --eqn kdv-1d --const 1.0 0.0 --train_steps 100000 --al_every 1000 --num_points 200 --mem_pts_total_budget 1000 --no-allow_ic --inverse --inverse_guess 0.5 -1.0 --select_anchors_every 1000 --anchor_budget 10"
    "--hidden_layers 2 --hidden_dim 32 --eqn diffhc-1d --train_steps 30000 --al_every 1000 --num_points 50 --mem_pts_total_budget 100"
)

# "--method kmeans --eig_weight_method nystrom_wo_N --eig_sampling pseudo --eig_memory --auto_al"
# "--method greedy --eig_weight_method nystrom_wo_N --eig_sampling pseudo --eig_memory --auto_al"
# "--method sampling --eig_weight_method nystrom_wo_N --eig_sampling pseudo --eig_memory --auto_al"

# "--method sampling --eig_weight_method residue --eig_sampling pseudo --auto_al"
# "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --auto_al"
# "--method sampling --eig_weight_method alignment --eig_sampling pseudo --auto_al"
# "--method greedy --eig_weight_method alignment_norm --eig_sampling pseudo --eig_memory --auto_al"
# "--method greedy --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"

algs=(
    "--method random --rand_method pseudo --rand_res_prop 0.5"
    "--method random --rand_method pseudo --rand_res_prop 0.8"
    "--method random --rand_method pseudo --rand_res_prop 0.95"
    "--method random --rand_method Hammersley --rand_res_prop 0.5"
    "--method random --rand_method Hammersley --rand_res_prop 0.8"
    "--method random --rand_method Hammersley --rand_res_prop 0.95"
    "--method random --rand_method Sobol --rand_res_prop 0.5"
    "--method random --rand_method Sobol --rand_res_prop 0.8"
    "--method random --rand_method Sobol --rand_res_prop 0.95"
    "--method residue --res_res_prop 0.5"
    "--method residue --res_res_prop 0.5 --res_all_types"
    "--method residue --res_res_prop 0.8"
    "--method residue --res_res_prop 0.8 --res_all_types"
    "--method residue --res_res_prop 0.95"
    "--method residue --res_res_prop 0.95 --res_all_types"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --auto_al"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --auto_al"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --eig_fixed_budget"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --eig_fixed_budget"
    "--method random --rand_method Hammersley --rand_res_prop 0.8 --autoscale_loss_w_bcs"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs"
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

                pdeargs="$pde $alg $loss $4"

                python ${current_dir}/al_pinn.py $pdeargs
                
            done

        done

    done

done