#!/bin/bash
#SBATCH --gpus 1
#SBATCH --partition=overcap
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --account=overcap

# "--signal=USR1@300" sends a signal to the job _step_ when it needs to exit.
# It has 5 minutes to do so, otherwise it is forcibly killed

# This srun is critical!  The signal won't be sent correctly otherwise


building_id=$1


echo "On node ${HOSTNAME}"
echo "Rendering texture maps for building ${building_id}"
echo "CUDA VISIBLE DEVICES ${CUDA_VISIBLE_DEVICES}"
nvidia-smi


conda activate salve-v8
source activate salve-v8

cd /srv/scratch/jlambert30/salve/salve
srun python -u scripts/render_dataset_bev.py \
    --num_processes 1 \
    --raw_dataset_dir ../zind_bridgeapi_2021_10_05 \
    --hypotheses_save_root ../zind_horizonnet_hypotheses_2022_09_29_test \
    --depth_save_root ../ZinD_Bridge_API_HoHoNet_Depth_Maps \
    --bev_save_root ../2022_09_29_zind_texture_map_renderings_test \
    --building_id $building_id
