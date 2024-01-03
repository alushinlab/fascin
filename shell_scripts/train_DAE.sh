#!/bin/bash
#SBATCH --partition=alus_a100
#SBATCH --error=./training_DAE_err_%j.txt
#SBATCH --output=./training_DAE_output_%j.txt
#SBATCH --nodelist=node[205]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mreynolds@rockefeller.edu

echo Starting at `date`
echo This is job $SLURM_JOB_ID
echo Running on `hostname`


for i in {0}; do 
    srun /rugpfs/fs0/cem/store/mreynolds/scripts/in_development/train_networks/train_DAE_UNet_3D.py \
	--input_noise_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject/noise_dir \
	--input_noiseless_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject/noiseless_dir \
	--proj_dim 64 \
	--numProjs 20000 \
	--tv_split 90 \
	--lr 0.0001 \
	--patience 3 \
	--epochs 30 \
	--batch_size 16 \
	--gpu_idx 2 \
	--preload_ram False \
	--output_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/testing_backproject/  &
done


wait
