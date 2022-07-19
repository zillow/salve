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
echo "Rendering BEV texture maps for ${building_id}"
echo "CUDA VISIBLE DEVICES ${CUDA_VISIBLE_DEVICES}"
nvidia-smi


conda activate salve-v1
source activate salve-v1
cd /srv/scratch/jlambert30/salve/HoHoNet
export PYTHONPATH=./
srun python -u ../salve/scripts/render_dataset_bev.py \
    --raw_dataset_dir /srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05 \
    --depth_save_root /srv/scratch/jlambert30/salve/ZinD_Bridge_API_HoHoNet_Depth_Maps \
    --hypotheses_save_root /srv/scratch/jlambert30/salve/2022_07_18_ZInD_alignment_hypotheses \
    --bev_save_root /srv/scratch/jlambert30/salve/2022_07_18_zind_bev_texture_maps \
    --num-processes 1 \
    --split test \
    --building_id $building_id