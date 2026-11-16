"""Exercise builder API selection without requiring CUDA or TensorRT."""
from types import SimpleNamespace

import pytest

from scripts.jetson.fast_foundation_build_trt import configure_tactic_search


def test_trt10_uses_levels_without_removed_flags():
    config = SimpleNamespace(builder_optimization_level=3)
    result = configure_tactic_search(SimpleNamespace(), config)
    assert config.builder_optimization_level == 2
    assert result == {'builder_optimization_level': 2, 'tactic_heuristic': False}
    configure_tactic_search(SimpleNamespace(), config, exhaustive=True)
    assert config.builder_optimization_level == 3
    configure_tactic_search(SimpleNamespace(), config, optimization_level=1)
    assert config.builder_optimization_level == 1


def test_trt85_keeps_original_heuristic_and_rejects_new_levels():
    calls = []
    config = SimpleNamespace(
        set_flag=lambda value: calls.append(('flag', value)),
        set_preview_feature=lambda key, value: calls.append(('preview', key, value)),
    )
    trt = SimpleNamespace(
        BuilderFlag=SimpleNamespace(ENABLE_TACTIC_HEURISTIC='heuristic'),
        PreviewFeature=SimpleNamespace(DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805='core'),
    )
    assert configure_tactic_search(trt, config)['tactic_heuristic']
    assert calls == [('flag', 'heuristic'), ('preview', 'core', True)]
    calls.clear()
    assert not configure_tactic_search(trt, config, exhaustive=True)['tactic_heuristic']
    assert calls == []
    with pytest.raises(ValueError, match='does not support'):
        configure_tactic_search(trt, config, optimization_level=2)
