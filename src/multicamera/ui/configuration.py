"""Dependency-aware invalidation for proposed camera configuration changes."""

from dataclasses import dataclass

from ..streaming.stream_manager import AuxiliaryCameraConfig, StereoPairConfig


@dataclass(frozen=True)
class ConfigurationImpact:
    reset_stereo: bool
    changed_pairs: frozenset[str]
    camera_ids: frozenset[str]
    reset_aux: bool
    streams_changed: bool


def pair_signature(pair: StereoPairConfig) -> tuple:
    return (pair.name, pair.left.camera_id, pair.left.url, pair.left.stream_type,
            pair.right.camera_id, pair.right.url, pair.right.stream_type)


def configuration_impact(old_board, new_board, old_model, new_model,
                         old_pairs: dict[str, StereoPairConfig], new_pairs: list[StereoPairConfig],
                         old_aux: AuxiliaryCameraConfig | None,
                         new_aux: AuxiliaryCameraConfig | None) -> ConfigurationImpact:
    proposed = {p.name: p for p in new_pairs}
    old_signatures = {name: pair_signature(pair) for name, pair in old_pairs.items()}
    new_signatures = {name: pair_signature(pair) for name, pair in proposed.items()}
    reset_stereo = old_board.config != new_board.config or old_model != new_model
    changed = {name for name in old_pairs.keys() | proposed.keys()
               if reset_stereo or old_signatures.get(name) != new_signatures.get(name)}
    while True:
        camera_ids = {cam.camera_id for pairs in (old_pairs, proposed)
                      for name, pair in pairs.items() if name in changed
                      for cam in (pair.left, pair.right)}
        dependencies = {name for pairs in (old_pairs, proposed) for name, pair in pairs.items()
                        if pair.left.camera_id in camera_ids or pair.right.camera_id in camera_ids}
        if dependencies <= changed:
            break
        changed.update(dependencies)
    old_ref = next((p.left.camera_id for p in old_pairs.values() if p.left.stream_type == "rgb"), None)
    new_ref = next((p.left.camera_id for p in proposed.values() if p.left.stream_type == "rgb"), None)
    aux_changed = ((old_aux.to_dict() if old_aux else None) !=
                   (new_aux.to_dict() if new_aux else None))
    return ConfigurationImpact(
        reset_stereo, frozenset(changed), frozenset(camera_ids),
        reset_stereo or aux_changed or old_ref != new_ref or old_ref in camera_ids,
        list(old_signatures.values()) != list(new_signatures.values()) or aux_changed,
    )
