"""Similar-day retrieval over the residual case pool."""
import numpy as np


def standardize(x, ntrain):
    """Standardize columns with statistics from training days only."""
    mu, sd = x[:ntrain].mean(0), x[:ntrain].std(0)
    return (x - mu) / np.where(sd < 1e-8, 1.0, sd)


def neighbors(keys, metric, queries, pool, k, rng=None, exclude=None, causal=True, gap=0):
    """Indices (len(queries), k) of the k most similar pool days for each query day.

    metric: "cosine", "euclidean" or "random".
    causal=True: only days strictly before the query (forecasting setting).
    causal=False: any pool day with |day - query| > gap (diagnostic only, not a forecast).
    exclude: optional bool mask of days removed from the pool.
    """
    if metric == "cosine":
        keys = keys / np.linalg.norm(keys, axis=1, keepdims=True).clip(1e-12)
    idx = np.arange(len(pool))
    out = np.empty((len(queries), k), dtype=int)
    for i, d in enumerate(queries):
        ok = pool & ((idx < d) if causal else (np.abs(idx - d) > gap))
        if exclude is not None:
            ok &= ~exclude
        cand = np.flatnonzero(ok)
        assert len(cand) >= k, f"pool too small for day {d}"
        if causal:
            assert cand.max() < d, "pool must contain only earlier days"
        if metric == "random":
            out[i] = rng.choice(cand, k, replace=False)
            continue
        if metric == "cosine":
            dist = -(keys[cand] @ keys[d])
        else:
            dist = ((keys[cand] - keys[d]) ** 2).sum(1)
        out[i] = cand[np.argsort(dist, kind="stable")[:k]]
    return out
