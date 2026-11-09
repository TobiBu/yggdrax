"""Static gradient audit of a traced program: which equations can carry a cotangent.

jaccpot's distributed-gradient audit (``docs/differentiable_fmm_distributed_audit.md``
there) found every blocker with two cheap probes before running a single gradient:
``jax.make_jaxpr(jax.grad(loss))``, and a LIVENESS SWEEP over the forward jaxpr.
This module is the sweep.

Starting from the differentiated inputs, a variable is *live* when its value
depends on them through a path a tangent can follow: through inexact (float)
values only, and never through ``stop_gradient``. Integer and boolean results end
a path (no tangent survives a cast to an integer), so a Morton code, a sort
permutation or a walk's pair list is never live, whatever it was computed from.
The sweep recurses into every sub-jaxpr (``shard_map``, ``jit``, ``scan``,
``while``, ``cond``, ``checkpoint``, custom-derivative calls), iterating loop
carries to a fixed point.

What the drivers' fixed-topology contract requires, and what the tests assert:

* no ``while`` loop has a live input. Reverse mode cannot cross a
  ``while_loop`` anyway, and in this pipeline every one of them is a topology loop
  (tree build, walk, chunked top-k, node max): a live input there would mean a
  topology decision is being differentiated.

Run with ``stop_at_stop_gradient=False`` it is the control: the same sweep with
the seam removed finds the topology loops live, which shows the sweep can see
them at all.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax.extend import core as jcore


class LiveWhile(NamedTuple):
    """A ``while`` equation reached by a live input."""

    where: str
    live_inputs: int


def _is_float(aval: Any) -> bool:
    dtype = getattr(aval, "dtype", None)
    return dtype is not None and jnp.issubdtype(dtype, jnp.inexact)


def _where(eqn: Any) -> str:
    try:
        from jax._src import source_info_util

        return source_info_util.summarize(eqn.source_info)
    except Exception:  # pragma: no cover - private API, best effort only
        return str(eqn.primitive)


def _open(j: Any) -> Any:
    """A ``Jaxpr`` from a ``Jaxpr`` or ``ClosedJaxpr``."""
    return j.jaxpr if hasattr(j, "jaxpr") and hasattr(j, "consts") else j


class _Sweep:
    def __init__(self, stop_at_stop_gradient: bool):
        self.stop = stop_at_stop_gradient
        self.whiles: list[LiveWhile] = []
        self.live_eqns: dict[str, int] = {}

    def run(self, jaxpr: Any, live_in: list[bool]) -> list[bool]:
        """Propagate liveness through ``jaxpr``; returns the outputs' liveness."""
        jaxpr = _open(jaxpr)
        live: set = set()
        for v, flag in zip(jaxpr.invars, live_in):
            if flag and _is_float(v.aval):
                live.add(v)

        def is_live(v):
            return not isinstance(v, jcore.Literal) and v in live

        for eqn in jaxpr.eqns:
            ins = [is_live(v) for v in eqn.invars]
            if not any(ins):
                continue
            name = eqn.primitive.name
            if name == "stop_gradient" and self.stop:
                continue
            outs = self._eqn(eqn, ins)
            if any(outs):
                self.live_eqns[name] = self.live_eqns.get(name, 0) + 1
            for v, flag in zip(eqn.outvars, outs):
                if flag and _is_float(v.aval):
                    live.add(v)
        return [is_live(v) for v in jaxpr.outvars]

    def _eqn(self, eqn: Any, ins: list[bool]) -> list[bool]:
        p = eqn.params
        name = eqn.primitive.name
        n_out = len(eqn.outvars)
        if name == "while":
            nc, nb = p["cond_nconsts"], p["body_nconsts"]
            body_consts = ins[nc : nc + nb]
            carry = list(ins[nc + nb :])
            while True:
                out = self.run(p["body_jaxpr"], body_consts + carry)
                new = [a or b for a, b in zip(carry, out)]
                if new == carry:
                    break
                carry = new
            self.whiles.append(LiveWhile(_where(eqn), sum(ins)))
            return carry
        if name == "scan":
            if "num_consts" in p:  # older jax (e.g. 0.8.3): counts as params
                nc, ncarry = p["num_consts"], p["num_carry"]
                consts = ins[:nc]
                carry = list(ins[nc : nc + ncarry])
                xs = ins[nc + ncarry :]
            else:  # newer jax (e.g. 0.11.2): FlatTree (consts, carry, xs)
                consts, carry, xs = (list(g) for g in p["ft_in"].update(ins).unpack())
                ncarry = len(carry)
            while True:
                out = self.run(p["jaxpr"], consts + carry + xs)
                new = [a or b for a, b in zip(carry, out[:ncarry])]
                if new == carry:
                    return carry + out[ncarry:]
                carry = new
        if name == "cond":
            outs = [False] * n_out
            for br in p["branches"]:
                outs = [a or b for a, b in zip(outs, self.run(br, ins[1:]))]
            return outs
        for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
            sub = p.get(key)
            if sub is not None and len(_open(sub).invars) == len(ins):
                return self.run(sub, ins)
        # an unknown primitive: any live input makes every float output live
        return [True] * n_out


def live_while_loops(
    jaxpr: Any, live_in: list[bool], *, stop_at_stop_gradient: bool = True
) -> tuple[list[LiveWhile], dict[str, int]]:
    """``while`` loops reached by a live input, and a count of live primitives.

    Parameters
    ----------
    jaxpr:
        ``jax.make_jaxpr(f)(*args)``.
    live_in:
        One flag per flattened input: the differentiated ones.
    stop_at_stop_gradient:
        ``False`` is the control sweep (the seam removed).

    Returns
    -------
    tuple
        ``(whiles, counts)``.
    """
    sweep = _Sweep(stop_at_stop_gradient)
    sweep.run(jaxpr, live_in)
    return sweep.whiles, sweep.live_eqns
