#!/bin/bash

#SBATCH -J siggame
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=4GB
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
# python3 -u ippo_ff.py WANDB_NOTES="Codename-R6 Rerun lr diff - quadrortx - Trying 10 agents with a reintroduction of env images" ENV_KWARGS.channel_ratio_fn="sigmoid-custom-cutoff-3500" ENV_KWARGS.sigmoid_offset=700 ENV_KWARGS.sigmoid_stretch=0.006 SPEAKER_TRAIN_SCHEDULE="off then on at 350" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 350 anneal to 2e-5 at 2000"
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R1c - L2 Reg Speakers - Trying 5 agents and looking for results with new iconicity probe" ENV_KWARGS.channel_ratio_fn="sigmoid-custom" ENV_KWARGS.sigmoid_offset=600 ENV_KWARGS.sigmoid_stretch=0.005 SPEAKER_TRAIN_SCHEDULE="off then on at 324" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 324 anneal to 2e-5 at 1500" L2_REG_COEF_LISTENER=0.0
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R3 - Speaker LR Boost and smoother channel ratio fn" ENV_KWARGS.channel_ratio_fn="sigmoid-custom" ENV_KWARGS.sigmoid_offset=800 ENV_KWARGS.sigmoid_stretch=0.003 SPEAKER_TRAIN_SCHEDULE="off then on at 324" LISTENER_LR_SCHEDULE="1e-4 jump to 2e-6 at 324" SPEAKER_LR_SCHEDULE="2e-4"
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R5f - More listeners - tighter channel ratio fn - negative whitesum penalty - higher speaker lr" ENV_KWARGS.channel_ratio_fn="sigmoid-custom" ENV_KWARGS.sigmoid_offset=690 ENV_KWARGS.sigmoid_stretch=0.005 ENV_KWARGS.speaker_whitesum_penalty_coef=-0.05 SPEAKER_TRAIN_SCHEDULE="off then on at 250" LISTENER_LR_SCHEDULE="1e-4 jump to 3e-5 at 250" SPEAKER_LR_SCHEDULE="1e-3"
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R6b - Tiny whitesum penalty - higher speaker lr" ENV_KWARGS.channel_ratio_fn="sigmoid-custom" ENV_KWARGS.sigmoid_offset=710 ENV_KWARGS.sigmoid_stretch=0.003 ENV_KWARGS.speaker_whitesum_penalty_coef=0.01 SPEAKER_TRAIN_SCHEDULE="off then on at 250" LISTENER_LR_SCHEDULE="1e-4 jump to 2e-5 at 250" SPEAKER_LR_SCHEDULE="6e-4"
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R7b - Annealing again - negative tiny whitesum penalty" ENV_KWARGS.channel_ratio_fn="sigmoid-custom" ENV_KWARGS.sigmoid_offset=710 ENV_KWARGS.sigmoid_stretch=0.003 ENV_KWARGS.speaker_whitesum_penalty_coef=-0.02 SPEAKER_TRAIN_SCHEDULE="off then on at 250" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 250 anneal to 1e-5 at 1500" SPEAKER_LR_SCHEDULE="2e-4"
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R8b - Tweaking 1901" ENV_KWARGS.channel_ratio_fn="sigmoid-custom" ENV_KWARGS.sigmoid_offset=760 ENV_KWARGS.sigmoid_stretch=0.003 ENV_KWARGS.speaker_whitesum_penalty_coef=0.0 SPEAKER_TRAIN_SCHEDULE="off then on at 300" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 300 anneal to 1e-5 at 2000" SPEAKER_LR_SCHEDULE="2e-4" L2_REG_COEF_LISTENER=1e-5 L2_REG_COEF_SPEAKER=1e-5
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R9d - Running with 1926 - Even Tighter channel ratio fn - Tiny whitesum penalty" ENV_KWARGS.channel_ratio_fn=sigmoid-custom ENV_KWARGS.sigmoid_offset=498 ENV_KWARGS.sigmoid_stretch=0.007 ENV_KWARGS.speaker_whitesum_penalty_coef=0.01 SPEAKER_TRAIN_SCHEDULE="off then on at 300" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 300 anneal to 2e-5 at 1500" SPEAKER_LR_SCHEDULE=2e-4 L2_REG_COEF_LISTENER=1e-5 L2_REG_COEF_SPEAKER=1e-5
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R10d - negative tiny whitesum penalty" ENV_KWARGS.channel_ratio_fn=sigmoid-custom ENV_KWARGS.sigmoid_offset=498 ENV_KWARGS.sigmoid_stretch=0.007 ENV_KWARGS.speaker_whitesum_penalty_coef=-0.01 SPEAKER_TRAIN_SCHEDULE="off then on at 300" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 300 anneal to 2e-5 at 1500" SPEAKER_LR_SCHEDULE=2e-4 L2_REG_COEF_LISTENER=1e-5 L2_REG_COEF_SPEAKER=1e-5
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R11b - Auto-centering and no penalties - 3 splines" ENV_KWARGS.channel_ratio_fn=sigmoid-custom ENV_KWARGS.sigmoid_offset=498 ENV_KWARGS.sigmoid_stretch=0.007 ENV_KWARGS.speaker_whitesum_penalty_coef=0.0 ENV_KWARGS.center_listener_obs=True ENV_KWARGS.speaker_action_dim=18 SPEAKER_TRAIN_SCHEDULE="off then on at 300" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 300 anneal to 2e-5 at 1500" SPEAKER_LR_SCHEDULE=2e-4 L2_REG_COEF_LISTENER=1e-5 L2_REG_COEF_SPEAKER=1e-5
# python3 -u ippo_ff.py WANDB_NOTES="Post-Hiatus-R12d - Negative Smaller Curve penalty" ENV_KWARGS.channel_ratio_fn=sigmoid-custom ENV_KWARGS.sigmoid_offset=498 ENV_KWARGS.sigmoid_stretch=0.007 ENV_KWARGS.speaker_whitesum_penalty_coef=0.0 ENV_KWARGS.speaker_curve_penalty_coef=-0.05 ENV_KWARGS.center_listener_obs=False ENV_KWARGS.speaker_action_dim=12 SPEAKER_TRAIN_SCHEDULE="off then on at 300" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 300 anneal to 2e-5 at 1500" SPEAKER_LR_SCHEDULE=2e-4 L2_REG_COEF_LISTENER=1e-5 L2_REG_COEF_SPEAKER=1e-5 WANDB_MODE="online"
python3 -u ippo_ff.py WANDB_NOTES="Post-Draft-Part1a-R3 - Asymmetric no penalties quicker speaker onset - Zero listener rewards" ENV_KWARGS.channel_ratio_fn=sigmoid-custom ENV_KWARGS.sigmoid_offset=400 ENV_KWARGS.sigmoid_stretch=0.007 ENV_KWARGS.sigmoid_height=0.6 ENV_KWARGS.speaker_assignment_method="arange" ENV_KWARGS.symmetric_rewards=False ENV_KWARGS.speaker_whitesum_penalty_coef=0.0 ENV_KWARGS.speaker_curve_penalty_coef=0.0 SPEAKER_TRAIN_SCHEDULE="off then on at 300" LISTENER_LR_SCHEDULE="1e-4 jump to 1e-6 at 300 anneal to 2e-5 at 1500" SPEAKER_LR_SCHEDULE=2e-4 L2_REG_COEF_LISTENER=1e-5 L2_REG_COEF_SPEAKER=1e-5 WANDB_MODE="online"
