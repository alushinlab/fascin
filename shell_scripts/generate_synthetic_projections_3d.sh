#!/bin/bash
#SBATCH --partition=alus_a100
#SBATCH --error=./projection_gen_err_%j.txt
#SBATCH --output=./projection_gen_output_%j.txt
#SBATCH --nodelist=node[205]
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mreynolds@rockefeller.edu

echo Starting at `date`
echo This is job $SLURM_JOB_ID
echo Running on `hostname`

for i in {0}; do 
    srun /rugpfs/fs0/cem/store/mreynolds/scripts/in_development/generate_synth_data_3d/projection_generator_3D.py \
        --input_mrc_dir /rugpfs/fs0/cem/store/mreynolds/scripts/in_development/generate_synth_data_3d/testing/mrc_library/ \
	--output_noise_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/synth_data/noise_dir \
	--output_noiseless_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/synth_data/noiseless_dir \
	--output_semMap_dir /rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/synth_data/semMap \
	--numProjs 100032 \
	--nProcs 48 \
	--box_len 64 \
	--alt_stdev 20 \
	--tx_low -25 \
	--tx_high 25 \
	--ty_low -25 \
	--ty_high 25 \
	--tz_low -25 \
	--tz_high 25 \
	--max_fil_num 1 \
	--is_bundle False \
	--kV 300 \
	--ampCont 10.0 \
	--angpix 7.80 \
	--bfact 0.0 \
	--cs 2.7 \
	--defoc_low 1.0 \
	--defoc_high 7.0 \
	--noise_amp 0.050 \
	--noise_stdev 0.010 \
	--use_empirical_model yes \
	--path_to_empirical_model /rugpfs/fs0/cem/store/mreynolds/scripts/in_development/generate_synth_data_3d/empirical_noise_models/ \
	--path_to_empirical_model_STD /rugpfs/fs0/cem/store/mreynolds/scripts/in_development/generate_synth_data_3d/empirical_noise_models/ \
	--lowpass_res 30 \
	--dil_rad 1 \
	--erod_rad 0  &
done


wait
