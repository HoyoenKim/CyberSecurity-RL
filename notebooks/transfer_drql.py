# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
# %%
"""Notebook demonstrating transfer learning capability of the
the Deep Recurrent Q-learning (DRQN) agent trained and evaluated on the chain
 environment of various sizes.
"""

# %%
import os
import sys
import logging
import gymnasium as gym
import torch

import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_drqn as drqn   # ★ DRQN 추가
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.agent_randomcredlookup as rca
import importlib
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import cyberbattle._env.cyberbattle_chain as cyberbattle_chain

importlib.reload(learner)
importlib.reload(cyberbattle_env)
importlib.reload(cyberbattle_chain)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# %matplotlib inline

# %%
torch.cuda.is_available()

# %%
cyberbattlechain_4 = gym.make(
    "CyberBattleChain-v0",
    size=4,
    attacker_goal=cyberbattle_env.AttackerGoal(own_atleast_percent=1.0)
).unwrapped
cyberbattlechain_10 = gym.make(
    "CyberBattleChain-v0",
    size=10,
    attacker_goal=cyberbattle_env.AttackerGoal(own_atleast_percent=1.0)
).unwrapped
cyberbattlechain_20 = gym.make(
    "CyberBattleChain-v0",
    size=20,
    attacker_goal=cyberbattle_env.AttackerGoal(own_atleast_percent=1.0)
).unwrapped

assert isinstance(cyberbattlechain_4, cyberbattle_env.CyberBattleEnv)
assert isinstance(cyberbattlechain_10, cyberbattle_env.CyberBattleEnv)
assert isinstance(cyberbattlechain_20, cyberbattle_env.CyberBattleEnv)

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
    maximum_node_count=22,
    identifiers=cyberbattlechain_10.identifiers,
)

# %% {"tags": ["parameters"]}
iteration_count = 9000
training_episode_count = 50
eval_episode_count = 10
plots_dir = "output/images"

# %%
os.makedirs(plots_dir, exist_ok=True)

# %% ---------------------------------------------------------
# Run Deep Recurrent Q-learning (DRQN) on size=10 environment
# ----------------------------------------------------------
best_drqn_learning_run_10 = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlechain_10,
    environment_properties=ep,
    learner=drqn.DeepQLearnerPolicy(   # ★ DQN → DRQN
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        learning_rate=0.01,
        seq_len=1,
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    # epsilon_multdecay=0.75,  # 0.999,
    epsilon_exponential_decay=5000,  # 10000
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="DRQN",
)

# %%
# Plot episode length (DRQN)
p.plot_episodes_length([best_drqn_learning_run_10])

# %%
if not os.path.exists("images"):
    os.mkdir("images")

# %% ---------------------------------------------------------
# Exploit policy learned by DRQN (no exploration, epsilon=0)
# ----------------------------------------------------------
drqn_exploit_run = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    ep,
    learner=best_drqn_learning_run_10["learner"],
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,  # exploit only
    render=False,
    render_last_episode_rewards_to=os.path.join(plots_dir, "drqn_transfer-chain10"),
    title="Exploiting DRQN",
    verbosity=Verbosity.Quiet,
)

# %%
# Random baseline
random_run = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    ep,
    learner=learner.RandomPolicy(),
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=1.0,  # purely random
    render=False,
    verbosity=Verbosity.Quiet,
    title="Random search",
)

# %% ---------------------------------------------------------
# Plot averaged cumulative rewards for DRQN vs Random vs DRQN-Exploit
# (state/action space는 기존 DQN 모델 정의 재활용)
# ----------------------------------------------------------
themodel = dqla.CyberBattleStateActionModel(ep)
p.plot_averaged_cummulative_rewards(
    all_runs=[best_drqn_learning_run_10, random_run, drqn_exploit_run],
    title=(
        f"Benchmark (DRQN) -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n"
        f"State: {[f.name() for f in themodel.state_space.feature_selection]} "
        f"({len(themodel.state_space.feature_selection)}\n"
        f"Action: abstract_action ({themodel.action_space.flat_size()})"
    ),
)

# %%
# plot cumulative rewards for all episodes (DRQN training on size=10)
p.plot_all_episodes(best_drqn_learning_run_10)

##################################################
# %%
# DRQN training on size=4 environment
best_drqn_4 = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlechain_4,
    environment_properties=ep,
    learner=drqn.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        learning_rate=0.01,
        seq_len=1,
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="DRQN (size=4)",
)

# %% ---------------------------------------------------------
# Transfer learning evaluation (DRQN trained on size=10 → eval on size=20)
# ----------------------------------------------------------
learner.transfer_learning_evaluation(
    environment_properties=ep,
    trained_learner=best_drqn_learning_run_10,   # ★ DRQN(10)
    eval_env=cyberbattlechain_20,
    eval_epsilon=0.0,  # exploit only; can alternate with exploration to help generalization
    eval_episode_count=eval_episode_count,
    iteration_count=iteration_count,
    benchmark_policy=rca.CredentialCacheExploiter(),
    benchmark_training_args={
        "epsilon": 0.90,
        "epsilon_exponential_decay": 10000,
        "epsilon_minimum": 0.10,
        "title": "Credential lookups (ϵ-greedy)",
    },
)

# DRQN trained on size=4 → eval on size=10
learner.transfer_learning_evaluation(
    environment_properties=ep,
    trained_learner=best_drqn_4,   # ★ DRQN(4)
    eval_env=cyberbattlechain_10,
    eval_epsilon=0.0,  # exploit Q-matrix only
    eval_episode_count=eval_episode_count,
    iteration_count=iteration_count,
    benchmark_policy=rca.CredentialCacheExploiter(),
    benchmark_training_args={
        "epsilon": 0.90,
        "epsilon_exponential_decay": 10000,
        "epsilon_minimum": 0.10,
        "title": "Credential lookups (ϵ-greedy)",
    },
)

# %% ---------------------------------------------------------
# DRQN trained on size=4 → eval on size=20
# ----------------------------------------------------------
learner.transfer_learning_evaluation(
    environment_properties=ep,
    trained_learner=best_drqn_4,
    eval_env=cyberbattlechain_20,
    eval_epsilon=0.0,  # exploit Q-matrix only
    eval_episode_count=eval_episode_count,
    iteration_count=iteration_count,
    benchmark_policy=rca.CredentialCacheExploiter(),
    benchmark_training_args={
        "epsilon": 0.90,
        "epsilon_exponential_decay": 10000,
        "epsilon_minimum": 0.10,
        "title": "Credential lookups (ϵ-greedy)",
    },
)
