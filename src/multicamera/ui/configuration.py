"""Dependency-aware invalidation for proposed camera configuration changes."""

from dataclasses import dataclass, replace

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


def continuation_impact(old_board, new_board, old_model, new_model, old_pairs, new_pairs,
                        old_aux, new_aux, calibrated_names):
    """Transport-only reuse, only after explicit same-device/lens confirmation."""
    proposed = {pair.name: pair for pair in new_pairs}
    old_reference = next((p.left.camera_id for p in old_pairs.values() if p.left.stream_type == 'rgb'), None)
    new_reference = next((p.left.camera_id for p in new_pairs if p.left.stream_type == 'rgb'), None)
    if old_reference != new_reference:
        raise ValueError('续采不能改变融合 RGB 参考相机；请保持原组顺序或新建会话')
    if old_board.config != new_board.config or old_model != new_model:
        raise ValueError('续采不能修改 RGB 标定板或镜头模型；请取消续采并重新标定')
    normalized = dict(old_pairs)
    for name in calibrated_names:
        old, new = old_pairs.get(name), proposed.get(name)
        if old is None or new is None or (
                old.left.camera_id, old.right.camera_id, old.left.stream_type, old.right.stream_type
                ) != (new.left.camera_id, new.right.camera_id,
                      new.left.stream_type, new.right.stream_type):
            raise ValueError(f'{name}: 续采须保留原左右 ID、顺序和类型；更换设备请重新标定')
        normalized[name] = replace(old, left=replace(old.left, url=new.left.url),
                                   right=replace(old.right, url=new.right.url))
    original = configuration_impact(old_board, new_board, old_model, new_model,
                                    old_pairs, new_pairs, old_aux, new_aux)
    normalized_aux = old_aux
    if old_aux is not None and new_aux is not None:
        old_data, new_data = old_aux.to_dict(), new_aux.to_dict()
        old_data.pop('url')
        new_data.pop('url')
        if old_data == new_data:
            normalized_aux = replace(old_aux, url=new_aux.url)
    preserved = configuration_impact(old_board, new_board, old_model, new_model,
                                     normalized, new_pairs, normalized_aux, new_aux)
    if preserved.changed_pairs & set(calibrated_names):
        raise ValueError('续采配置影响了已标定相机组，请保留原组及左右绑定')
    return replace(preserved, streams_changed=original.streams_changed)


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
