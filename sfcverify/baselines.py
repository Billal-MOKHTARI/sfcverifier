"""Data-driven baselines learned from normal traces.

Both baselines are deterministic and model-free: they memorise what was observed during a
training period, like control-flow/state mining approaches.

  TransitionWhitelist  alarm when the pair (S(t-1), S(t)) was never observed in training
  StateWhitelist       alarm when the configuration S(t) was never observed in training

Unlike the structural verifier they need training data, and they raise false alarms on legal
behaviour that the training period did not cover (e.g. other interleavings of parallel
branches).
"""

from __future__ import annotations


class TransitionWhitelist:
    def __init__(self, training_masks: list[int]) -> None:
        self.pairs = {(p, c) for p, c in zip(training_masks, training_masks[1:]) if p != c}
        self.states = set(training_masks)

    def is_valid(self, prev: int, curr: int) -> bool:
        if prev == curr:
            return curr in self.states
        return (prev, curr) in self.pairs


class StateWhitelist:
    def __init__(self, training_masks: list[int]) -> None:
        self.states = set(training_masks)

    def is_valid(self, prev: int, curr: int) -> bool:
        return curr in self.states
