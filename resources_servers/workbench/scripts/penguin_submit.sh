# ----- PARAMETERS -----
# WANDB_API_KEY, SLURM_ACCOUNT, SLURM_PARTITION, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION



# /lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/examples/penguin/grpo_workbench_qwen3_4binstruct.yaml

# --container-image=
# ----- CONSTANTS -----

CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/llmservice/users/bxyu/cache/containers/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-2aea5add.squashfs


read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}



HF_HOME=.cache/ \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    cluster.num_nodes=$NUM_ACTOR_NODES \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    checkpointing.checkpoint_dir=results/$EXP_NAME \
    ++grpo.num_prompts_per_step=8 \
    ++grpo.max_num_steps=200 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    ++max_val_samples=544 \
    ++keep_top_k=10 \
    ++save_period=1 \
    ++val_period=1 \
    $@
EOF

echo -e "Running command:\n$COMMAND"

# Not sure why this is necessary, but ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
