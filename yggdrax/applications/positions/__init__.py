"""Move a particle set until its clustering statistics match a target's.

Every coordinate of a catalogue in a periodic box is a parameter; the loss
compares its ξ (traced soft pair counts, window rule) and its kNN-CDFs with a
target catalogue's, and an ``optax`` optimiser descends on it with gradients
through both tree estimators. See :mod:`yggdrax.applications.positions.matching`
for the loss, the rebuild cadence and the capacity handling.
"""

from yggdrax.applications.positions.matching import (
    LossTerms,
    MatchSpec,
    MatchState,
    MatchStatistics,
    OptimiseResult,
    PositionMatcher,
    cap_displacement,
    default_spec,
    lbfgs_positions,
    optimise_positions,
)

__all__ = [
    "LossTerms",
    "MatchSpec",
    "MatchState",
    "MatchStatistics",
    "OptimiseResult",
    "PositionMatcher",
    "cap_displacement",
    "default_spec",
    "lbfgs_positions",
    "optimise_positions",
]
