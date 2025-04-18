#!/bin/bash
# Script to run particle lenia illumination with time-series data collection for causal blanket analysis

python illuminate_wrapper.py \
    --seed=0 \
    --save_dir="./data/particle_lenia_illumination" \
    --substrate="plenia" \
    --n_child=32 \
    --pop_size=256 \
    --n_iters=10 \
    --sigma=0.1 \
    --k_nbrs=2 \
    --save_time_series \
    --time_sampling_rate=4 \
    --cb_eval_subset=32 \
    --rollout_steps=512 \
    --state_in_bc_calc=True
