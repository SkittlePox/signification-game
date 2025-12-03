#!/bin/bash -l
#SBATCH -p swarm_a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00


module load conda/python3
conda activate siggame

python icon_probe.py ENV_DATASET=cifar100 ENV_KWARGS.num_classes=100 WANDB_MODE=offline MODEL_NAME_PREFIX="probe-cifar100-5k-1k-60e-adam-"

python3 -u ippo_ff.py WANDB_NOTES="Post-Draft-Part2-R14 - cifar10b just listeners 5000" +dataset=cifar10b ENV_KWARGS.channel_ratio_fn="0.0" LISTENER_LR_SCHEDULE="1e-4" SPEAKER_TRAIN_SCHEDULE="off" UPDATE_EPOCHS=2000 ENV_NUM_DATAPOINTS=5000 WANDB_MODE=offline PICKLE_FINAL_AGENTS=True PROBE_MODEL_NAME="probe-cifar10b-5k-1k-60e-adam-1d21"

python3 -u ippo_ff.py WANDB_NOTES="Post-Draft-Part2-R13 - cifar10b tom agents pr 1.0 no penalties speaker n search anneal with larger speaker l2 norm term" +dataset=cifar10b ENV_KWARGS.channel_ratio_fn="0.5" ENV_KWARGS.speaker_curve_penalty_coef=0.0 ENV_KWARGS.speaker_whitesum_penalty_coef=0.0 ENV_KWARGS.speaker_assignment_method="random" ENV_KWARGS.agent_inferential_mode_fn="1.0" LISTENER_N_SAMPLES=6 SPEAKER_N_SEARCH="10 jump to 10 at 300 anneal to 1 at 750" LISTENER_PR_WEIGHT=1.0 L2_REG_COEF_SPEAKER=1e-2 LISTENER_LR_SCHEDULE="1e-5" SPEAKER_LR_SCHEDULE="1e-4" SPEAKER_TRAIN_SCHEDULE="on" PRETRAINED_LISTENERS="agents-cifar10b-None-2000e-5000dp-b324" UPDATE_EPOCHS=3000 ENV_NUM_DATAPOINTS=5000 WANDB_MODE=offline PICKLE_FINAL_AGENTS=False LOG_TOM_SPEAKER_EXAMPLES=True PROBE_MODEL_NAME="probe-cifar10b-5k-1k-60e-adam-1d21"

#python3 -u ippo_ff.py WANDB_NOTES="Post-Draft-Part2-R13 - cifar10b tom agents pr 1.0 no penalties speaker n search anneal with larger speaker l2 norm term" +dataset=cifar10b ENV_KWARGS.channel_ratio_fn="0.5" ENV_KWARGS.speaker_curve_penalty_coef=0.0 ENV_KWARGS.speaker_whitesum_penalty_coef=0.0 ENV_KWARGS.speaker_assignment_method="random" ENV_KWARGS.agent_inferential_mode_fn="1.0" LISTENER_N_SAMPLES=6 SPEAKER_N_SEARCH="10 jump to 10 at 300 anneal to 1 at 750" LISTENER_PR_WEIGHT=1.0 L2_REG_COEF_SPEAKER=1e-2 LISTENER_LR_SCHEDULE="1e-5" SPEAKER_LR_SCHEDULE="1e-4" SPEAKER_TRAIN_SCHEDULE="on" PRETRAINED_LISTENERS="agents-cifar10b-woven-glade-2275-2000e-5000dp-c6fb" UPDATE_EPOCHS=3000 ENV_NUM_DATAPOINTS=5000 WANDB_MODE=online PICKLE_FINAL_AGENTS=False LOG_TOM_SPEAKER_EXAMPLES=True
