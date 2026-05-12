#!/usr/bin/env python3
"""
Generate a parameter sweep CSV for peregrine.py.

Sweep types (--sweep-type):
- random: sample N architecture combinations. The draw strategy is controlled
  by --sampling-mode (see below).
- ofat: one-factor-at-a-time over every value in PARAM_VALUES for that factor.
- default: a single row matching DEFAULT_PARAM_VALUES.
- tweak: one row per parameter; all values at default except that parameter,
  set to the first value in PARAM_VALUES[key] that differs from the default
  (keys with no alternative are skipped).

Random sampling modes (--sampling-mode, random sweeps only):
- grid: uniform iid over the raw grid, ignoring feasibility. Distinct by
  default (mixed-radix index decode; safe for the ~10^21 design space).
- valid-uniform: rejection-free uniform iid over the feasible region defined
  by ``is_valid_combination`` (mirrors ``optimize_hw_config.is_valid``). Uses
  a factorized inverse-CDF sampler; each draw is O(1) in the space size.
- valid-lhs: Latin Hypercube over the same factorized conditional CDFs.
  Better space-filling than iid for a given N; recommended when the sweep
  feeds an ML regression model.
- valid-cost-stratified: draws an oversampled uniform-valid pool, bins by
  ``hardware_cost`` quantile, and samples within bins. Concentrates training
  data along the cost-efficient frontier where the Pareto front lives.

Both ``is_valid_combination`` and ``hardware_cost`` are mirrored here from
peregrine/ml_model/optimize_hw_config.py so the sampler and the downstream
Pareto optimizer agree on feasibility and cost weightings.
"""

import argparse
import random

import numpy as np
import pandas as pd

# Predefined values per parameter. Modify as needed.
PARAM_VALUES = {
    "int_reg_issue_width": list(range(1, 8 + 1)),
    "int_mult_div_issue_width": list(range(1, 8 + 1)),
    "fp_reg_issue_width": list(range(1, 8 + 1)),
    "fp_mult_div_issue_width": list(range(1, 8 + 1)),
    "read_port_issue_width": list(range(1, 8 + 1)),
    "rdwr_port_issue_width": list(range(1, 8 + 1)),
    "simd_unit_issue_width": [1],
    "fetch_width": list(range(1, 12 + 1)),
    "decode_width": list(range(1, 12 + 1)),
    "rename_width": list(range(1, 12 + 1)),
    "commit_width": list(range(1, 12 + 1)),
    "rob_size": list(range(1, 1024 + 1)),
    "lq_entries": list(range(1, 256 + 1)),
    "sq_entries": list(range(1, 256 + 1)),
    "branch_predictor": ["local", "tage"],
    "l1d_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l1i_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l2_size": ["512KiB", "1MiB", "2MiB", "4MiB"],
    "max_icache_fills": list(range(1, 32 + 1)),
    "stride_prefetcher_degree": [0, 4],
}

# Constraints mirrored from peregrine/ml_model/optimize_hw_config.py so the
# sampler and the downstream Pareto optimizer agree on feasibility.
ROB_HEADROOM = 8
ISSUE_WIDTH_RENAME_RATIO = 2.0


def _parse_size_to_kb(s):
    """Convert a cache size string (e.g. "32KiB", "1MiB") to KB as int."""
    import re

    s = str(s).strip()
    m = re.match(r"^(\d+)\s*KiB$", s, re.I)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)\s*MiB$", s, re.I)
    if m:
        return int(m.group(1)) * 1024
    raise ValueError(f"Unrecognized cache size: {s!r}")


def hardware_cost(combos):
    """Relative area/cost estimate per configuration.

    Mirror of ``hardware_cost`` in peregrine/ml_model/optimize_hw_config.py
    (same weightings, same formula). Vectorized via pandas so cost-stratified
    sampling can evaluate large oversample pools in one call.
    """
    if not combos:
        return []

    df = pd.DataFrame(combos)
    fu_cost = (
        1.0 * df["int_reg_issue_width"]
        + 5.0 * df["int_mult_div_issue_width"]
        + 3.0 * df["fp_reg_issue_width"]
        + 8.0 * df["fp_mult_div_issue_width"]
        + 3.0 * df["read_port_issue_width"]
        + 4.0 * df["rdwr_port_issue_width"]
    )
    total_issue_width = (
        df["int_reg_issue_width"]
        + df["int_mult_div_issue_width"]
        + df["fp_reg_issue_width"]
        + df["fp_mult_div_issue_width"]
        + df["read_port_issue_width"]
        + df["rdwr_port_issue_width"]
    )
    issue_network_cost = 0.4 * total_issue_width**2
    pipe_cost = (
        2.0 * df["fetch_width"]
        + 3.0 * df["decode_width"]
        + 4.0 * df["rename_width"]
        + 2.0 * df["commit_width"]
    )
    rob_cost = 0.8 * df["rob_size"]
    lsq_cost = 2.5 * (df["lq_entries"] + df["sq_entries"])
    mshr_cost = 3.0 * df["max_icache_fills"]
    bp_core_map = {"local": 10.0, "tage": 100.0}
    bp_core = df["branch_predictor"].astype(str).str.lower().map(bp_core_map)
    if bp_core.isna().any():
        bad = df.loc[bp_core.isna(), "branch_predictor"].unique().tolist()
        raise ValueError(f"Unrecognized branch_predictor values: {bad}")
    bp_cost = bp_core + 30.0
    prefetch_cost = (df["stride_prefetcher_degree"] > 0).astype(float) * 15.0

    cpu_cost = (
        fu_cost
        + issue_network_cost
        + pipe_cost
        + rob_cost
        + lsq_cost
        + mshr_cost
        + bp_cost
        + prefetch_cost
    )
    l1i_kb = df["l1i_size"].map(_parse_size_to_kb)
    l1d_kb = df["l1d_size"].map(_parse_size_to_kb)
    l2_kb = df["l2_size"].map(_parse_size_to_kb)
    memory_cost = (5.0 * (l1i_kb + l1d_kb) + 1.0 * l2_kb) * 0.1
    return (cpu_cost + memory_cost).tolist()


def is_valid_combination(combo):
    """True iff ``combo`` satisfies the O3 pipeline / queue feasibility rules.

    Mirrors ``is_valid`` in optimize_hw_config.py. Kept scalar (dict-input) so
    the sampling path below can filter row-at-a-time during rejection sampling.
    """
    fetch = combo["fetch_width"]
    decode = combo["decode_width"]
    rename = combo["rename_width"]
    commit = combo["commit_width"]
    rob = combo["rob_size"]
    lq = combo["lq_entries"]
    sq = combo["sq_entries"]
    rp = combo["read_port_issue_width"]
    rw = combo["rdwr_port_issue_width"]
    ir = combo["int_reg_issue_width"]
    im = combo["int_mult_div_issue_width"]
    fr = combo["fp_reg_issue_width"]
    fm = combo["fp_mult_div_issue_width"]

    mem_ports = rp + rw
    total_issue = ir + im + fr + fm + rp + rw

    return (
        decode <= fetch
        and rename <= decode
        and commit <= rename
        and rob >= ROB_HEADROOM * commit
        and lq >= mem_ports
        and sq >= mem_ports
        and total_issue <= ISSUE_WIDTH_RENAME_RATIO * rename
    )


DEFAULT_PARAM_VALUES = {
    "int_reg_issue_width": 2,
    "int_mult_div_issue_width": 2,
    "fp_reg_issue_width": 2,
    "fp_mult_div_issue_width": 2,
    "read_port_issue_width": 2,
    "rdwr_port_issue_width": 2,
    "simd_unit_issue_width": 1,
    "fetch_width": 8,
    "decode_width": 8,
    "rename_width": 8,
    "commit_width": 8,
    "rob_size": 192,
    "lq_entries": 32,
    "sq_entries": 32,
    "branch_predictor": "local",
    "l1d_size": "32KiB",
    "l1i_size": "32KiB",
    "l2_size": "512KiB",
    "max_icache_fills": 4,
    "stride_prefetcher_degree": 4,
}

# Fixed order: same as itertools.product(*value_lists), last varies fastest
PARAM_KEYS = sorted(PARAM_VALUES.keys())


def first_non_default_value(key):
    """First entry in PARAM_VALUES[key] that differs from DEFAULT_PARAM_VALUES[key], or None."""
    d = DEFAULT_PARAM_VALUES[key]
    for v in PARAM_VALUES[key]:
        if v != d:
            return v
    return None


def total_combinations(param_values):
    """Total number of combinations (product of all value list sizes)."""
    n = 1
    for k in PARAM_KEYS:
        n *= len(param_values[k])
    return n


def _compute_strides(param_values):
    """Strides for mixed-radix decode: strides[i] = product of sizes[i+1:]."""
    sizes = [len(param_values[k]) for k in PARAM_KEYS]
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return sizes, strides


def index_to_combination(index, param_values, sizes=None, strides=None):
    """
    Decode a linear index into a parameter dict using mixed-radix.
    Order matches itertools.product(*value_lists) with PARAM_KEYS: last varies fastest.
    Pass precomputed sizes, strides (from _compute_strides) when decoding many indices
    to avoid recomputation and to keep working with huge totals.
    """
    value_lists = [param_values[k] for k in PARAM_KEYS]
    if sizes is None or strides is None:
        sizes, strides = _compute_strides(param_values)
    digits = [(index // strides[i]) % sizes[i] for i in range(len(sizes))]
    return dict(
        zip(PARAM_KEYS, [value_lists[i][d] for i, d in enumerate(digits)])
    )


# When n >= total we would return all indices; only materialize if total is small
_MAX_INDICES_MATERIALIZE = 10**9


def sample_random_indices(total, n):
    """
    Return n distinct random indices in [0, total) without materializing range(total).
    Safe for arbitrarily large total (e.g. > 10^20); uses randrange and a set.
    When n >= total, returns all indices only if total <= _MAX_INDICES_MATERIALIZE.
    """
    if n >= total:
        if total <= _MAX_INDICES_MATERIALIZE:
            return list(range(total))
        raise ValueError(
            f"total combinations ({total}) is very large; "
            "num_combinations must be less than total (use -n to sample a subset)."
        )
    chosen = []
    seen = set()
    while len(chosen) < n:
        idx = random.randrange(total)
        if idx not in seen:
            seen.add(idx)
            chosen.append(idx)
    return chosen


def _fu_sum_distributions(fu_vals):
    """Return (dist_by_k, min_sum_k) with dist_by_k[k][s] = the number of
    k-tuples over ``fu_vals`` summing to ``s``, for k in 1..4."""

    def convolve(a, b):
        out = {}
        for sa, ca in a.items():
            for vb in b:
                out[sa + vb] = out.get(sa + vb, 0) + ca
        return out

    dist1 = {v: 1 for v in fu_vals}
    dist2 = convolve(dist1, fu_vals)
    dist3 = convolve(dist2, fu_vals)
    dist4 = convolve(dist3, fu_vals)
    dist_by_k = {1: dist1, 2: dist2, 3: dist3, 4: dist4}
    min_sum_k = {k: min(d) for k, d in dist_by_k.items()}
    return dist_by_k, min_sum_k


def _precompute_valid_sampler(param_values):
    """Precompute the factorized weight tables used by the direct sampler.

    The feasibility predicate (see is_valid_combination) factorizes as:

        (fetch >= decode >= rename) AND
        (commit <= rename, rob >= ROB_HEADROOM * commit) AND
        (rp + rw = M, lq >= M, sq >= M,
         ir + im + fr + fm + M <= ISSUE_WIDTH_RENAME_RATIO * rename)

    Conditioning on ``rename`` makes the three groups independent, so the
    count of feasible configs factors per-rename into
        fd_count(r) * cr_weight(r) * fu_weight(r).
    We precompute each factor (and per-leaf sampling buckets) once so that
    draws are O(1) in the design-space size.
    """
    fu_cols = (
        "int_reg_issue_width",
        "int_mult_div_issue_width",
        "fp_reg_issue_width",
        "fp_mult_div_issue_width",
        "read_port_issue_width",
        "rdwr_port_issue_width",
    )
    fu_vals = param_values[fu_cols[0]]
    for col in fu_cols[1:]:
        if param_values[col] != fu_vals:
            raise ValueError(
                f"Direct sampler requires all FU domains identical; "
                f"{col!r} differs from {fu_cols[0]!r}."
            )

    dist_by_k, min_sum_k = _fu_sum_distributions(fu_vals)
    dist2 = dist_by_k[2]
    dist4 = dist_by_k[4]
    min_s4 = min_sum_k[4]

    fetch_grid = param_values["fetch_width"]
    decode_grid = param_values["decode_width"]
    rename_grid = param_values["rename_width"]
    commit_grid = param_values["commit_width"]
    rob_grid = param_values["rob_size"]
    lq_grid = param_values["lq_entries"]
    sq_grid = param_values["sq_entries"]

    # LQ/SQ choices keyed by M = rp + rw (must satisfy lq, sq >= M).
    lq_values_by_M = {M: [v for v in lq_grid if v >= M] for M in dist2}
    sq_values_by_M = {M: [v for v in sq_grid if v >= M] for M in dist2}

    # (rp, rw) pairs grouped by their sum M.
    pairs_by_M: dict = {}
    for rp in fu_vals:
        for rw in fu_vals:
            pairs_by_M.setdefault(rp + rw, []).append((rp, rw))

    # ROB choices keyed by commit value.
    rob_values_by_c = {
        c: [v for v in rob_grid if v >= ROB_HEADROOM * c] for c in commit_grid
    }
    rob_count_by_c = {c: len(v) for c, v in rob_values_by_c.items()}

    # 4-tuple-sum sampling buckets keyed by remaining budget (= budget - M).
    max_budget = int(ISSUE_WIDTH_RENAME_RATIO * max(rename_grid))
    sum_lists_by_remaining: dict = {}
    for remaining in range(min_s4, max_budget + 1):
        sums, ws = [], []
        for s_val in range(min_s4, remaining + 1):
            cnt = dist4.get(s_val, 0)
            if cnt > 0:
                sums.append(s_val)
                ws.append(cnt)
        if sums:
            sum_lists_by_remaining[remaining] = (sums, ws)

    per_rename: dict = {}
    for r in rename_grid:
        budget = int(ISSUE_WIDTH_RENAME_RATIO * r)

        # M = rp + rw distribution weighted by its contribution to fu_weight(r).
        m_values, m_weights = [], []
        for M, pair_count in dist2.items():
            remaining = budget - M
            if remaining < min_s4:
                continue
            lq_c = len(lq_values_by_M[M])
            sq_c = len(sq_values_by_M[M])
            # cum4(remaining) = count of 4-tuples summing to <= remaining.
            sums_ws = sum_lists_by_remaining.get(remaining)
            cum4 = sum(sums_ws[1]) if sums_ws is not None else 0
            w = pair_count * lq_c * sq_c * cum4
            if w > 0:
                m_values.append(M)
                m_weights.append(w)
        fu_weight = sum(m_weights)

        # Pipeline (fetch, decode) pairs with fetch >= decode >= r.
        valid_fd = [
            (f, d) for f in fetch_grid for d in decode_grid if f >= d >= r
        ]

        # Commit candidates (c <= r) weighted by the number of feasible rob values.
        commit_values = [c for c in commit_grid if c <= r]
        commit_weights = [rob_count_by_c[c] for c in commit_values]
        cr_weight = sum(commit_weights)

        total = fu_weight * len(valid_fd) * cr_weight
        per_rename[r] = {
            "budget": budget,
            "m_values": m_values,
            "m_weights": m_weights,
            "valid_fd": valid_fd,
            "commit_values": commit_values,
            "commit_weights": commit_weights,
            "total": total,
        }

    rename_values = list(rename_grid)
    rename_weights = [per_rename[r]["total"] for r in rename_values]
    total_constrained = sum(rename_weights)
    if total_constrained == 0:
        raise ValueError(
            "No valid configurations exist under the current PARAM_VALUES "
            "and feasibility constraints."
        )

    # Independent (unconstrained) parameters contribute a flat multiplier.
    constrained = set(fu_cols) | {
        "fetch_width",
        "decode_width",
        "rename_width",
        "commit_width",
        "rob_size",
        "lq_entries",
        "sq_entries",
    }
    indep_params = [k for k in param_values if k not in constrained]
    indep_multiplier = 1
    for p in indep_params:
        indep_multiplier *= len(param_values[p])

    return {
        "fu_vals": fu_vals,
        "dist_by_k": dist_by_k,
        "rename_values": rename_values,
        "rename_weights": rename_weights,
        "per_rename": per_rename,
        "rob_values_by_c": rob_values_by_c,
        "pairs_by_M": pairs_by_M,
        "lq_values_by_M": lq_values_by_M,
        "sq_values_by_M": sq_values_by_M,
        "sum_lists_by_remaining": sum_lists_by_remaining,
        "indep_params": indep_params,
        "total_valid": total_constrained * indep_multiplier,
    }


def _inverse_cdf_pick(values, weights, u):
    """Inverse-CDF lookup: given uniform ``u`` in [0, 1), return values[i]
    where i is the smallest index with cum_weight[i]/total >= u."""
    total = sum(weights)
    target = u * total
    acc = 0.0
    for v, w in zip(values, weights):
        acc += w
        if acc > target:
            return v
    return values[-1]


def _sample_tuple_with_sum_icdf(s, k, dist_by_k, fu_vals, us):
    """Uniform k-tuple over ``fu_vals`` summing to ``s``, driven by ``us`` —
    a length-(k-1) array of uniforms in [0, 1)."""
    tup = []
    remaining = s
    for step in range(k - 1):
        prev = dist_by_k[k - 1 - step]
        cand, weights = [], []
        for v in fu_vals:
            c = prev.get(remaining - v, 0)
            if c > 0:
                cand.append(v)
                weights.append(c)
        pick = _inverse_cdf_pick(cand, weights, us[step])
        tup.append(pick)
        remaining -= pick
    tup.append(remaining)
    return tup


# Axes driven by inverse-CDF lookups from a uniform vector u of length
# N_SAMPLER_AXES. LHS allocates one column per axis; iid uniform just draws
# independent uniforms. Kept as a constant so callers can build LHS plans.
# Axes in order:
#   0 rename_width
#   1 (fetch, decode) pair
#   2 commit
#   3 rob
#   4 M = rp + rw
#   5 (rp, rw) pair given M
#   6 S = ir+im+fr+fm
#   7,8,9 tuple-with-sum picks (ir, im, fr), fm determined
#   10 lq
#   11 sq
# Then one uniform per independent parameter.
_N_STRUCTURED_AXES = 12


def _structured_draw_from_uniforms(u_row, state, param_values):
    """Map a uniform vector ``u_row`` (length structured_axes + indep_params)
    to a single feasible configuration dict."""
    dist_by_k = state["dist_by_k"]
    fu_vals = state["fu_vals"]

    r = _inverse_cdf_pick(
        state["rename_values"], state["rename_weights"], u_row[0]
    )
    info = state["per_rename"][r]

    fd_pairs = info["valid_fd"]
    f, d = fd_pairs[int(u_row[1] * len(fd_pairs)) % len(fd_pairs)]

    c = _inverse_cdf_pick(
        info["commit_values"], info["commit_weights"], u_row[2]
    )
    rob_choices = state["rob_values_by_c"][c]
    rob = rob_choices[int(u_row[3] * len(rob_choices)) % len(rob_choices)]

    M = _inverse_cdf_pick(info["m_values"], info["m_weights"], u_row[4])
    pairs = state["pairs_by_M"][M]
    rp, rw = pairs[int(u_row[5] * len(pairs)) % len(pairs)]

    remaining = info["budget"] - M
    sums, weights = state["sum_lists_by_remaining"][remaining]
    S = _inverse_cdf_pick(sums, weights, u_row[6])
    ir, im, fr, fm = _sample_tuple_with_sum_icdf(
        S, 4, dist_by_k, fu_vals, u_row[7:10]
    )

    lq_choices = state["lq_values_by_M"][M]
    sq_choices = state["sq_values_by_M"][M]
    lq = lq_choices[int(u_row[10] * len(lq_choices)) % len(lq_choices)]
    sq = sq_choices[int(u_row[11] * len(sq_choices)) % len(sq_choices)]

    combo = {
        "fetch_width": f,
        "decode_width": d,
        "rename_width": r,
        "commit_width": c,
        "rob_size": rob,
        "lq_entries": lq,
        "sq_entries": sq,
        "int_reg_issue_width": ir,
        "int_mult_div_issue_width": im,
        "fp_reg_issue_width": fr,
        "fp_mult_div_issue_width": fm,
        "read_port_issue_width": rp,
        "rdwr_port_issue_width": rw,
    }
    for k_idx, p in enumerate(state["indep_params"]):
        grid = param_values[p]
        combo[p] = grid[
            int(u_row[_N_STRUCTURED_AXES + k_idx] * len(grid)) % len(grid)
        ]
    return combo


def _draw_uniform_matrix(n, d, rng):
    """iid uniform [0, 1) matrix, shape (n, d), driven by stdlib ``rng``."""
    return np.array([[rng.random() for _ in range(d)] for _ in range(n)])


def _draw_lhs_matrix(n, d, rng):
    """Latin Hypercube uniform [0, 1) matrix, shape (n, d). Each column is a
    permutation of the n equal-probability strata, jittered uniformly within
    each stratum so downstream inverse-CDF lookups remain unbiased."""
    strata = (
        np.arange(n)[:, None]
        + np.random.default_rng(rng.randrange(2**63)).random((n, d))
    ) / n
    out = np.empty_like(strata)
    rng_np = np.random.default_rng(rng.randrange(2**63))
    for j in range(d):
        out[:, j] = rng_np.permutation(strata[:, j])
    return out


def sample_valid_combinations(param_values, n, rng=None, *, lhs=False):
    """Rejection-free sampler over configurations satisfying
    ``is_valid_combination``.

    Factorizes the feasibility constraints analytically so each draw is O(1)
    in the ~10^21 design-space size. When ``lhs`` is False (default), draws
    are iid uniform over the feasible region. When ``lhs`` is True, draws are
    mapped from a Latin Hypercube design in [0, 1)^d through the feasibility
    inverse-CDFs — preserving uniformity of each marginal while giving
    better space-filling than iid for a given N.
    """
    if n <= 0:
        return [], {
            "total_valid": 0,
            "n": 0,
            "mode": "lhs" if lhs else "uniform",
        }
    if rng is None:
        rng = random.Random()

    state = _precompute_valid_sampler(param_values)
    d = _N_STRUCTURED_AXES + len(state["indep_params"])

    if lhs:
        u = _draw_lhs_matrix(n, d, rng)
    else:
        u = _draw_uniform_matrix(n, d, rng)

    combos = [
        _structured_draw_from_uniforms(u[i], state, param_values)
        for i in range(n)
    ]
    return combos, {
        "total_valid": state["total_valid"],
        "n": n,
        "mode": "lhs" if lhs else "uniform",
    }


def sample_valid_cost_stratified(
    param_values, n, rng=None, *, oversample=8, n_bins=None, lhs=True
):
    """Cost-stratified sampler over the feasible region.

    Draws a pool of ``oversample * n`` valid configs (via LHS by default for
    space-fill, or iid uniform if ``lhs=False``), bins them into ``n_bins``
    equal-WIDTH buckets of ``hardware_cost`` between the pool's min and max,
    and uniformly subsamples within bins. Equal-width (not equal-count) binning
    is the key: the raw valid-uniform cost distribution is bell-shaped and
    under-samples the low- and high-cost tails where the Pareto front lives.
    Equal-width bins guarantee every part of the cost range (including the
    sparse tails) is represented in the training set.

    ``n_bins`` defaults to min(n, 20). Each bin contributes approximately
    ``n / n_bins`` rows; empty bins (no pool points fell into that cost band)
    redistribute their share across the non-empty bins proportionally.
    """
    if n <= 0:
        return [], {"total_valid": 0, "n": 0, "mode": "cost_stratified"}
    if rng is None:
        rng = random.Random()

    if n_bins is None:
        n_bins = min(n, 20)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    pool_n = max(oversample * n, n_bins * 4)
    pool, pool_stats = sample_valid_combinations(
        param_values, pool_n, rng=rng, lhs=lhs
    )
    costs = np.asarray(hardware_cost(pool))

    edges = np.linspace(costs.min(), costs.max(), n_bins + 1)
    # np.digitize places points into 1..n_bins; shift to 0-indexed and clamp
    # the top edge inclusive.
    bin_idx = np.clip(np.digitize(costs, edges[1:-1]), 0, n_bins - 1)

    bucket_members = [np.where(bin_idx == b)[0] for b in range(n_bins)]
    non_empty = [b for b in range(n_bins) if len(bucket_members[b]) > 0]
    if not non_empty:
        raise RuntimeError("Cost-stratified pool produced no populated bins.")

    # Distribute draws evenly across non-empty bins; spread any remainder
    # across the first few in order so totals sum exactly to n.
    per_bin = {b: n // len(non_empty) for b in non_empty}
    for b in non_empty[: n - sum(per_bin.values())]:
        per_bin[b] += 1

    rng_np = np.random.default_rng(rng.randrange(2**63))
    selected_indices = []
    for b in non_empty:
        bucket = bucket_members[b]
        want = per_bin[b]
        picks = rng_np.choice(bucket, size=want, replace=want > len(bucket))
        selected_indices.extend(int(x) for x in picks)

    combos = [pool[i] for i in selected_indices]
    return combos, {
        "total_valid": pool_stats["total_valid"],
        "n": len(combos),
        "mode": "cost_stratified",
        "pool_n": pool_n,
        "n_bins": n_bins,
        "n_bins_populated": len(non_empty),
        "cost_min": float(costs.min()),
        "cost_max": float(costs.max()),
    }


# ---------------------------------------------------------------------------
# Dispatcher: unifies the sampling modes for callers.
# ---------------------------------------------------------------------------

SAMPLING_MODES = (
    "grid",
    "valid-uniform",
    "valid-lhs",
    "valid-cost-stratified",
)


def sample_combinations(param_values, n, mode, rng=None, **kwargs):
    """Dispatch to the sampling mode requested.

    Modes (see module docstring for full detail):
      - ``"grid"``: uniform iid over the raw grid, ignoring feasibility
        (distinct draws via mixed-radix index decode). ``kwargs`` accepts
        ``with_replacement`` (default False).
      - ``"valid-uniform"``: uniform iid over the feasible region.
      - ``"valid-lhs"``: Latin Hypercube over the feasible region.
      - ``"valid-cost-stratified"``: cost-quantile-stratified draws within
        the feasible region. ``kwargs`` accepts ``oversample`` and ``n_bins``.

    Returns ``(combinations, stats)``. ``stats`` always contains ``mode`` and
    ``n``; mode-specific fields vary.
    """
    if mode not in SAMPLING_MODES:
        raise ValueError(
            f"Unknown sampling mode {mode!r}; expected one of {SAMPLING_MODES}"
        )
    if rng is None:
        rng = random.Random()

    if mode == "grid":
        with_replacement = kwargs.pop("with_replacement", False)
        if kwargs:
            raise TypeError(f"Unexpected kwargs for grid mode: {list(kwargs)}")
        total = total_combinations(param_values)
        sizes, strides = _compute_strides(param_values)
        if with_replacement:
            indices = [rng.randrange(total) for _ in range(n)]
        else:
            n_unique = min(n, total)
            # sample_random_indices uses module-level random; seed it so the
            # draw is reproducible relative to the caller's rng.
            random.seed(rng.randrange(2**63))
            indices = sample_random_indices(total, n_unique)
            if n_unique < n:
                indices.extend(
                    rng.randrange(total) for _ in range(n - n_unique)
                )
        combos = [
            index_to_combination(k, param_values, sizes=sizes, strides=strides)
            for k in indices
        ]
        return combos, {"mode": "grid", "n": len(combos), "total": total}

    if mode == "valid-uniform":
        if kwargs:
            raise TypeError(
                f"Unexpected kwargs for valid-uniform: {list(kwargs)}"
            )
        return sample_valid_combinations(param_values, n, rng=rng, lhs=False)

    if mode == "valid-lhs":
        if kwargs:
            raise TypeError(f"Unexpected kwargs for valid-lhs: {list(kwargs)}")
        return sample_valid_combinations(param_values, n, rng=rng, lhs=True)

    # valid-cost-stratified
    return sample_valid_cost_stratified(param_values, n, rng=rng, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter sweep CSV for peregrine.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="param_sweep.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "-t",
        "--sweep-type",
        type=str,
        default="random",
        choices=["random", "ofat", "default", "tweak"],
        help=(
            "random: sample N combos; ofat: every value of each param alone (many rows); "
            "default: one row of defaults; tweak: one row per param, only that param off-default."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-combinations",
        type=int,
        default=2**14,
        help=(
            "Number of combination indices to sample for random sweeps only. "
            "Ignored for ofat, default, and tweak."
        ),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=262,
        help="Random seed for random sweeps only; ignored otherwise.",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="grid",
        choices=list(SAMPLING_MODES),
        help=(
            "For random sweeps, which sampling strategy to use: "
            "'grid' (uniform iid over the raw grid; ignores feasibility), "
            "'valid-uniform' (uniform iid over the feasible region), "
            "'valid-lhs' (Latin Hypercube over the feasible region), or "
            "'valid-cost-stratified' (cost-quantile stratified draws — "
            "concentrates samples along the cost-efficient frontier). "
            "Ignored for non-random sweep types."
        ),
    )
    parser.add_argument(
        "--cost-stratified-oversample",
        type=int,
        default=8,
        help=(
            "Oversample factor for valid-cost-stratified mode: draws "
            "factor*N candidates, bins by hardware_cost, subsamples to N."
        ),
    )
    parser.add_argument(
        "--cost-stratified-bins",
        type=int,
        default=None,
        help=(
            "Number of cost-quantile bins for valid-cost-stratified mode. "
            "Defaults to min(N, 20)."
        ),
    )
    args = parser.parse_args()

    if args.sweep_type == "random":
        rng = random.Random(args.seed)
        total = total_combinations(PARAM_VALUES)
        n = min(args.num_combinations, total)

        mode_kwargs = {}
        if args.sampling_mode == "valid-cost-stratified":
            mode_kwargs["oversample"] = args.cost_stratified_oversample
            mode_kwargs["n_bins"] = args.cost_stratified_bins

        combinations, stats = sample_combinations(
            PARAM_VALUES, n, mode=args.sampling_mode, rng=rng, **mode_kwargs
        )
        if args.sampling_mode == "grid":
            print(
                f"Total combinations: {total:.2e}. Sampled {stats['n']} "
                "grid points (iid uniform, no feasibility filter)."
            )
        elif args.sampling_mode == "valid-cost-stratified":
            print(
                f"Total combinations: {total:.2e}; valid subspace: "
                f"{stats['total_valid']:.2e} "
                f"({stats['total_valid'] / total:.2%}). "
                f"Cost-stratified sampling: pool={stats['pool_n']} "
                f"bins={stats['n_bins']} "
                f"cost∈[{stats['cost_min']:.1f}, {stats['cost_max']:.1f}] "
                f"→ {stats['n']} configs."
            )
        else:
            print(
                f"Total combinations: {total:.2e}; valid subspace: "
                f"{stats['total_valid']:.2e} "
                f"({stats['total_valid'] / total:.2%}). "
                f"Sampled {stats['n']} valid configs via {stats['mode']}."
            )
    elif args.sweep_type == "ofat":
        combinations = []
        for key in PARAM_KEYS:
            defaulted = dict(DEFAULT_PARAM_VALUES)
            for value in PARAM_VALUES[key]:
                combo = dict(defaulted)
                combo[key] = value
                combinations.append(combo)
        print(
            f"OFAT sweep over {len(PARAM_KEYS)} parameters, "
            f"{len(combinations)} total combinations. "
            "Arguments --num-combinations and --seed are ignored for OFAT sweeps."
        )
    elif args.sweep_type == "default":
        combinations = [dict(DEFAULT_PARAM_VALUES)]
        print(
            "Single default baseline row (--num-combinations and --seed ignored)."
        )
    elif args.sweep_type == "tweak":
        combinations = []
        skipped = []
        for key in PARAM_KEYS:
            alt = first_non_default_value(key)
            if alt is None:
                skipped.append(key)
                continue
            row = dict(DEFAULT_PARAM_VALUES)
            row[key] = alt
            combinations.append(row)
        print(
            f"Tweak sweep: {len(combinations)} rows (one non-default value per parameter). "
            f"--num-combinations and --seed ignored."
        )
        if skipped:
            print(
                f"Skipped {len(skipped)} parameter(s) with no value != default in PARAM_VALUES: "
                f"{', '.join(skipped)}"
            )

    # One row per combination
    df = pd.DataFrame(combinations)
    df = df[PARAM_KEYS]
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
