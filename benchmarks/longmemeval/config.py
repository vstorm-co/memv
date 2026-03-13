"""Named MemoryConfig presets for LongMemEval benchmark ablations."""

from __future__ import annotations

from memv.config import MemoryConfig

CONFIGS: dict[str, MemoryConfig] = {
    "default": MemoryConfig(),
    # Fast: skips predict-calibrate, dedup, and merging. For iteration speed only —
    # results are NOT comparable to 'default' config.
    "fast": MemoryConfig(
        max_statements_for_prediction=0,
        enable_knowledge_dedup=False,
        enable_episode_merging=False,
    ),
    "no_predict_calibrate": MemoryConfig(max_statements_for_prediction=0),
    "no_segmentation": MemoryConfig(use_legacy_segmentation=True, segmentation_threshold=9999),
    "no_dedup": MemoryConfig(enable_knowledge_dedup=False, enable_episode_merging=False),
}


def get_config(name: str) -> MemoryConfig:
    """Get a named config preset.

    Args:
        name: One of: default, fast, no_predict_calibrate, no_segmentation, no_dedup.

    Returns:
        MemoryConfig for the named preset.
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {', '.join(CONFIGS)}")
    return CONFIGS[name]
