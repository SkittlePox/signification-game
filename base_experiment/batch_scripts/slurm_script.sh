#!/bin/bash

#SBATCH -J siggame
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C quadrortx
#SBATCH -o siggame_job_%j.o
#SBATCH -e siggame_job_%j.e
#SBATCH --mail-type=END
#SBATCH --mail-user=benjamin_spiegel@brown.edu

echo $LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

source /oscar/home/bspiegel/envs/jax.venv/bin/activate
cd /oscar/home/bspiegel/signification-game/base_experiment/
# python3 -u ippo_ff.py WANDB_NOTES="Codename-R1" ENV_KWARGS.sigmoid_offset=800 ENV_KWARGS.sigmoid_stretch=0.005 SPEAKER_TRAIN_SCHEDULE="off then on at 300" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 300 anneal to 1e-5 at 2000"
# python3 -u ippo_ff.py WANDB_NOTES="Codename-R4 - Trying 20 agents" ENV_KWARGS.sigmoid_offset=700 ENV_KWARGS.sigmoid_stretch=0.006 SPEAKER_TRAIN_SCHEDULE="off then on at 350" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 350 anneal to 1e-5 at 2000"
# python3 -u ippo_ff.py WANDB_NOTES="Codename-R3" ENV_KWARGS.sigmoid_offset=700 ENV_KWARGS.sigmoid_stretch=0.006 SPEAKER_TRAIN_SCHEDULE="off then on at 350" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 350 anneal to 1e-5 at 2000"
python3 -u ippo_ff.py WANDB_NOTES="Codename-R6 Rerun lr diff - quadrortx - Trying 10 agents with a reintroduction of env images" ENV_KWARGS.channel_ratio_fn="sigmoid-custom-cutoff-3500" ENV_KWARGS.sigmoid_offset=700 ENV_KWARGS.sigmoid_stretch=0.006 SPEAKER_TRAIN_SCHEDULE="off then on at 350" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 350 anneal to 2e-5 at 2000"
