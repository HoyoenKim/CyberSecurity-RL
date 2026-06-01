# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Smoke tests for the fixed DRQN agent (see CLAUDE.md §7-8 / §8).

These cover the two riskiest pieces of the DRQN fix without needing a full Gym
environment, so they run fast in CI (`pytest cyberbattle`):
  1. EpisodeReplayMemory pads episodes shorter than seq_len and returns a correct
     validity mask (previously short episodes silently disabled training).
  2. The masked Huber loss is finite and gradients flow (padding excluded).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cyberbattle.agents.baseline.agent_drqn import EpisodeReplayMemory


def _push(mem: EpisodeReplayMemory, i: int, done: bool, dim: int) -> None:
    mem.push(
        torch.randn(1, dim),                               # state
        torch.tensor([[i % 3]], dtype=torch.long),         # action
        torch.randn(1, dim),                               # next_state
        torch.tensor([1.0 if done else 0.0]),              # reward
        torch.tensor([1.0 if done else 0.0]),              # done
    )


def test_replay_padding_and_mask() -> None:
    dim, seq, batch = 5, 8, 32
    mem = EpisodeReplayMemory(capacity=10000, seq_len=seq)
    for i in range(20):            # long episode (>= seq_len)
        _push(mem, i, i == 19, dim)
    for i in range(3):             # short episode (< seq_len) -> must be padded + masked
        _push(mem, i, i == 2, dim)

    # sample() returns tensors on the agent's device (cuda if available); move to cpu so
    # these logic assertions are device-agnostic (pass on both GPU hosts and CPU-only CI).
    state, action, _reward, _next_state, done, mask = [t.cpu() for t in mem.sample(batch)]
    assert state.shape == (batch, seq, dim)
    assert action.shape == (batch, seq, 1)
    assert mask.shape == (batch, seq)
    for b in range(batch):
        valid = int(mask[b].sum().item())
        assert valid in (3, seq)
        # valid steps come first, padding after
        assert torch.equal(mask[b], torch.cat([torch.ones(valid), torch.zeros(seq - valid)]))
        if valid < seq:            # padded steps: done=1 and zero state (excluded from loss/bootstrap)
            assert bool(torch.all(done[b, valid:] == 1.0))
            assert bool(torch.all(state[b, valid:] == 0.0))


def test_masked_loss_grad_flows() -> None:
    dim, n_act, seq, batch = 5, 4, 8, 16
    mem = EpisodeReplayMemory(capacity=10000, seq_len=seq)
    for i in range(12):            # one episode, length 12 >= seq_len
        _push(mem, i, i == 11, dim)
    state, action, reward, next_state, done, mask = [t.cpu() for t in mem.sample(batch)]

    net = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, n_act))
    q = net(state).gather(2, action.long()).squeeze(2)            # [B, T]
    with torch.no_grad():
        target = reward + 0.9 * net(next_state).max(2)[0] * (1.0 - done)
    elementwise = F.smooth_l1_loss(q, target, reduction="none")   # [B, T]
    loss = (elementwise * mask).sum() / mask.sum().clamp_min(1.0)
    loss.backward()

    assert bool(torch.isfinite(loss))
    first_param = next(net.parameters())
    assert first_param.grad is not None
    assert bool(torch.isfinite(first_param.grad).all())
    assert float(first_param.grad.abs().sum()) > 0.0
