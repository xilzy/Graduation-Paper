"""Average-rank aggregation for S1 process evaluation.

Given per-method mean values for each metric, rank methods on every metric
(rank 1 = best, honouring the higher/lower-is-better direction) and report the
average rank per method. Lower average rank = better overall. This is the
single S1 score chosen in MASTER_PLAN.md section 5.4.
"""
import numpy as np
import pandas as pd

from .fusion_metrics import HIGHER_IS_BETTER


def average_rank_table(method_means: dict, metrics=None):
    """
    method_means: {method_name: {metric: mean_value}}
    metrics:      optional ordered list of metric names to rank on
                  (defaults to the union present, intersected with known direction)
    Returns a DataFrame indexed by method with one column per metric (the rank),
    plus 'AvgRank' (mean rank) and 'value::<metric>' raw-value columns,
    sorted by AvgRank ascending.
    """
    methods = list(method_means.keys())
    if metrics is None:
        seen = []
        for m in methods:
            for k in method_means[m]:
                if k in HIGHER_IS_BETTER and k not in seen:
                    seen.append(k)
        metrics = seen

    values = pd.DataFrame(
        {mt: [method_means[m].get(mt, np.nan) for m in methods] for mt in metrics},
        index=methods,
    )

    ranks = pd.DataFrame(index=methods)
    for mt in metrics:
        col = values[mt]
        ascending = not HIGHER_IS_BETTER.get(mt, True)   # rank 1 = best
        ranks[mt] = col.rank(ascending=ascending, method="average")

    ranks["AvgRank"] = ranks[metrics].mean(axis=1)
    for mt in metrics:
        ranks[f"value::{mt}"] = values[mt]
    return ranks.sort_values("AvgRank")
