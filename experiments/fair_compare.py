"""Fair, seeded DQN vs DRQN comparison on CyberBattleSim.

Addresses review issues (see CLAUDE.md §7):
  - 7-2 unfair comparison: identical exploration schedule + hyperparameters for both agents
  - 7-3 no seeds/single eval: multiple seeds, mean±std, per-eval-episode owned-node metric
  - 7-5 train cost: train_every throttles optimize frequency (same for both -> still fair)

Usage:
  python experiments/fair_compare.py [scenario] [seeds_csv]
    scenario : chain | defender    (default: defender)
    seeds_csv: e.g. 0,1,2          (default: 0,1,2)
  Env overrides for quick smoke runs: TRAIN_EP, ITERS, EVAL_EP
"""
import os, sys, time, json, random
import numpy as np
import torch
import gymnasium as gym
from typing import cast
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_drqn as drqn
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
from cyberbattle._env.cyberbattle_env import AttackerGoal, DefenderConstraint, CyberBattleEnv
from cyberbattle._env.defender import ScanAndReimageCompromisedMachines

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "defender"
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
SIZE = 10
TRAIN_EVERY = 4
# IDENTICAL config + schedule for both agents (fair comparison)
COMMON = dict(gamma=0.015, replay_memory_size=10000, target_update=10,
              batch_size=256, learning_rate=0.01, train_every=TRAIN_EVERY)
EPS = dict(epsilon=0.9, epsilon_exponential_decay=5000, epsilon_minimum=0.1)
TRAIN_EP = int(os.environ.get("TRAIN_EP", 50))
ITERS = int(os.environ.get("ITERS", 9000))
EVAL_EP = int(os.environ.get("EVAL_EP", 5))


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_env():
    if SCENARIO == "defender":
        e = gym.make("CyberBattleChain-v0", size=SIZE,
                     attacker_goal=AttackerGoal(own_atleast=0, own_atleast_percent=1.0),
                     defender_constraint=DefenderConstraint(maintain_sla=0.80),
                     defender_agent=ScanAndReimageCompromisedMachines(probability=0.6, scan_capacity=2, scan_frequency=5))
    elif SCENARIO == "toyctf":
        e = gym.make("CyberBattleToyCtf-v0")
    else:  # chain
        e = gym.make("CyberBattleChain-v0", size=SIZE)
    return cast(CyberBattleEnv, e.unwrapped)


def owned_total(ge):
    net = ge.environment.network
    owned = sum(1 for _, d in net.nodes(data=True) if d['data'].agent_installed)
    return owned, net.number_of_nodes()


def maker_for(name):
    if name == "DQN":
        return lambda ep: dqla.DeepQLearnerPolicy(ep=ep, **COMMON)
    return lambda ep: drqn.DeepQLearnerPolicy(ep=ep, seq_len=8, **COMMON)


def train_and_eval(name, seed):
    seed_all(seed)
    ge = make_env()
    ge.reset(seed=seed)
    try:
        ge.action_space.seed(seed)
    except Exception:
        pass
    ep = w.EnvironmentBounds.of_identifiers(maximum_node_count=22, maximum_total_credentials=22, identifiers=ge.identifiers)
    t0 = time.time()
    tr = learner.epsilon_greedy_search(
        cyberbattle_gym_env=ge, environment_properties=ep, learner=maker_for(name)(ep),
        episode_count=TRAIN_EP, iteration_count=ITERS, **EPS,
        verbosity=Verbosity.Quiet, render=False, plot_episodes_length=False, title="%s-s%d" % (name, seed))
    train_t = time.time() - t0
    trained = tr["learner"]
    owned_l, reward_l, len_l, tot = [], [], [], None
    for k in range(EVAL_EP):
        ev = learner.epsilon_greedy_search(
            ge, ep, learner=trained, episode_count=1, iteration_count=ITERS,
            epsilon=0.0, epsilon_minimum=0.0, verbosity=Verbosity.Quiet, render=False,
            plot_episodes_length=False, title="%s-s%d-ev%d" % (name, seed, k))
        o, tot = owned_total(ge)
        rr = (ev.get('all_episodes_rewards') or [[]])[-1]
        owned_l.append(o)
        reward_l.append(round(sum(rr), 1))
        len_l.append(len(rr))
    return dict(seed=seed, train_t=round(train_t, 1), total_nodes=tot,
                owned=owned_l, reward=reward_l, lengths=len_l,
                owned_mean=round(float(np.mean(owned_l)), 2), reward_mean=round(float(np.mean(reward_l)), 1))


def main():
    results = {}
    for name in ["DQN", "DRQN"]:
        runs = []
        for s in SEEDS:
            print(">>> %s seed=%d ..." % (name, s), flush=True)
            r = train_and_eval(name, s)
            print("    done: train %ss owned=%s reward_mean=%s" % (r['train_t'], r['owned'], r['reward_mean']), flush=True)
            runs.append(r)
        om = [x['owned_mean'] for x in runs]
        rm = [x['reward_mean'] for x in runs]
        results[name] = dict(runs=runs,
                             owned_mean=round(float(np.mean(om)), 2), owned_std=round(float(np.std(om)), 2),
                             reward_mean=round(float(np.mean(rm)), 1), reward_std=round(float(np.std(rm)), 1))

    out = dict(scenario=SCENARIO, size=SIZE, seeds=SEEDS, train_every=TRAIN_EVERY, common=COMMON, eps=EPS,
               train_ep=TRAIN_EP, iters=ITERS, eval_ep=EVAL_EP, results=results)
    os.makedirs(os.path.expanduser("~/hykim_ect/results"), exist_ok=True)
    outpath = os.path.expanduser("~/hykim_ect/results/fair_%s.json" % SCENARIO)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    tot = results["DQN"]["runs"][0]["total_nodes"]
    print("\n========== FAIR SWEEP (%s, size=%d, seeds=%s, train_every=%d) ==========" % (SCENARIO, SIZE, SEEDS, TRAIN_EVERY))
    for name in ["DQN", "DRQN"]:
        R = results[name]
        print("%-5s: owned %s±%s / %s | reward %s±%s | per-seed owned=%s" % (
            name, R['owned_mean'], R['owned_std'], tot, R['reward_mean'], R['reward_std'],
            [x['owned_mean'] for x in R['runs']]))
    print("saved:", outpath)
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
