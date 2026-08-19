"""Fusion evaluation metrics (Python port of the MATLAB suite in evaluation/).

All metrics operate on 2-D grayscale arrays with values in [0, 255] (float).
Convention for "reference-pair" metrics: arguments are (A, B, F) where A, B are
the two source images and F is the fused result.

See knowledge/paper/MASTER_PLAN.md section 5 for the metric selection rationale.
"""
from .fusion_metrics import (
    en, sd, sf, ag, mi, ssim_pair, ms_ssim_pair, qabf, vif_pair,
    scd, nabf, cc, psnr_pair,
    mi_split, ssim_split,
    ssim_hm, mi_hm, qabf_hm, vif_hm, ms_ssim_hm, balance,
    qabf_split, vif_split, ms_ssim_split,
    func_corr, func_sal,
    CORE_METRICS, DIAGNOSTIC_METRICS, RANK_METRICS, HIGHER_IS_BETTER,
    FIDELITY_AXIS, QUALITY_AXIS, FUNCTION_AXIS,
    compute_all,
)
from .aggregate import average_rank_table

__all__ = [
    "en", "sd", "sf", "ag", "mi", "ssim_pair", "ms_ssim_pair", "qabf",
    "vif_pair", "scd", "nabf", "cc", "psnr_pair", "mi_split", "ssim_split",
    "ssim_hm", "mi_hm", "qabf_hm", "vif_hm", "ms_ssim_hm", "balance",
    "qabf_split", "vif_split", "ms_ssim_split", "func_corr", "func_sal",
    "CORE_METRICS", "DIAGNOSTIC_METRICS", "RANK_METRICS", "HIGHER_IS_BETTER",
    "FIDELITY_AXIS", "QUALITY_AXIS", "FUNCTION_AXIS",
    "compute_all", "average_rank_table",
]
