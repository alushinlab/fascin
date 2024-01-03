#!/bin/bash
#SBATCH --partition=alus_a100
#SBATCH --error=./training_FCN_err_%j.txt
#SBATCH --output=./training_FCN_output_%j.txt
#SBATCH --nodelist=node[205]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mreynolds@rockefeller.edu

echo Starting at `date`
echo This is job $SLURM_JOB_ID
echo Running on `hostname`


for i in {0}; do 
    srun /rugpfs/fs0/cem/store/mreynolds/scripts/in_development/train_networks/train_FCN_for_semseg_3D.py \
	--input_noise_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject_rot90/noise_dir \
	--input_semseg_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject_rot90/semMap \
	--trained_DAE_path /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/testing_backproject_corrWedge_adv/adversarial_trained_DAE_epoch_000.h5 \
	--proj_dim 64 \
	--numProjs 10000 \
	--tv_split 90 \
	--lr 0.00001 \
	--patience 3 \
	--epochs 10 \
	--batch_size 16 \
	--gpu_idx 3 \
	--preload_ram False \
	--output_dir ./FCN_training_epoch000_try2/ &
done


wait
