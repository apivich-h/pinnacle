#!/bin/bash

current_dir=$( dirname -- "$0"; )
echo $current_dir

pdes=(
    "--hidden_layers 6 --hidden_dim 64 --nn laaf --eqn conv-1d --const 5.0"
    "--hidden_layers 4 --hidden_dim 32 --eqn kdv-1d --const 1.0 0.0 --al_every 5000"
    "--hidden_layers 6 --hidden_dim 64 --nn laaf --eqn fd-2d --const 1.0 0.01 --no-allow_ic --anc_measurable_idx 0 1 --inverse --inverse_guess 0 0 --al_every 1000 --select_anchors_every 1000 --anchor_budget 30"
    "--hidden_layers 8 --nn laaf --hidden_dim 32 --eqn eik1-3d --no-allow_ic --anc_measurable_idx 0 --train_steps 100000 --al_every 1000 --select_anchors_every 1000 --anchor_budget 5"
)


algs=(
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 100 --mem_pts_total_budget 300"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 100 --mem_pts_total_budget 500"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 200 --mem_pts_total_budget 1000"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 500 --mem_pts_total_budget 1000"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 100 --mem_pts_total_budget 300"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 100 --mem_pts_total_budget 500"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 200 --mem_pts_total_budget 1000"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --num_points 500 --mem_pts_total_budget 1000"
    "--method residue --res_res_prop 0.8 --res_all_types --mem_pts_total_budget 300"
    "--method residue --res_res_prop 0.8 --res_all_types --mem_pts_total_budget 500"
    "--method residue --res_res_prop 0.8 --res_all_types --mem_pts_total_budget 1000"
    "--method residue --res_res_prop 0.8 --res_all_types --mem_pts_total_budget 300 --res_unlimited_colloc"
    "--method residue --res_res_prop 0.8 --res_all_types --mem_pts_total_budget 500 --res_unlimited_colloc"
    "--method residue --res_res_prop 0.8 --res_all_types --mem_pts_total_budget 1000 --res_unlimited_colloc"
    "--method random --rand_method Hammersley --rand_res_prop 0.8 --mem_pts_total_budget 300"
    "--method random --rand_method Hammersley --rand_res_prop 0.8 --mem_pts_total_budget 1000"
    "--method random --rand_method Hammersley --rand_res_prop 0.8 --mem_pts_total_budget 10000"
    "--method random --rand_method Hammersley --rand_res_prop 0.8 --mem_pts_total_budget 50000"
    "--method random --rand_method Hammersley --rand_res_prop 0.8 --mem_pts_total_budget 100000"
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

                pdeargs="$pde $alg $loss"

                python ${current_dir}/al_pinn_timing.py $pdeargs
                
            done

        done

    done

done