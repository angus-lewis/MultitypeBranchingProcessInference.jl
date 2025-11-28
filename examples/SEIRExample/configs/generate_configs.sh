#!/bin/bash

julia --project=../../. generate_configs.jl seninrconfig.yaml seninr_nsteps_experiments.yaml obs_prob_experiments.yaml R0_experiments.yaml sigma_experiments.yaml

julia --project=../../. generate_configs.jl seirconfig.yaml seir_nsteps_experiments.yaml obs_prob_experiments.yaml R0_experiments.yaml sigma_experiments.yaml