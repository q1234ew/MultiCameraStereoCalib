"""Read-only preparation of a historical calibration's offline workspace."""

from dataclasses import dataclass
from pathlib import Path

from ..board.charuco_board import CharucoBoard
from ..calibration.models import CameraModel, MultiCameraRig
from ..calibration.offline_aux import is_legacy_offline_pair, validate_session_rig
from ..calibration.planar import PatternBoardConfig
from ..streaming.stream_manager import AuxiliaryCameraConfig, CameraConfig, StereoPairConfig


@dataclass
class SessionRestorePlan:
    rig: MultiCameraRig
    board: CharucoBoard
    model: CameraModel
    pairs: list[StereoPairConfig]
    auxiliary: AuxiliaryCameraConfig | None
    legacy_offline: bool


def inspect_session_files(session):
    """Inventory only: detection/decoding stays in the existing background importer."""
    root = session.session_dir
    missing, warnings, ready = [], [], []
    result = session.result_path
    if result.is_file():
        ready.append(f'双目结果：{result.relative_to(root)}')
    else:
        missing.append('双目结果 JSON：results/rig.json、有效最新批次或 calibration_all_json.json')
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    for pair in session.metadata.get('pairs', []):
        folder = root / 'images' / pair['name']
        sides = {side: {p.stem[len(side) + 1:]: p for p in folder.glob(side + '_*')
                        if p.is_file() and p.suffix.lower() in extensions} for side in ('left', 'right')}
        common = set(sides['left']) & set(sides['right'])
        if common:
            ready.append(f'{pair["name"]} 双目原图：{len(common)} 对，等待后台角点检查')
        for side, opposite in (('left', 'right'), ('right', 'left')):
            for key in sorted(set(sides[side]) - set(sides[opposite])):
                missing.append(f'images/{pair["name"]}/{sides[side][key].name} 缺少 {opposite}_{key}.*')
    if not (root / 'images').is_dir() or not any((root / 'images').rglob('*.png')):
        warnings.append('未发现双目原图；已有有效双目结果时，不阻止继续融合采集')
    images = {'rgb_left': {}, 'aux': {}}
    for path in (root / 'multimodal').glob('*'):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        for prefix in images:
            if path.stem.startswith(prefix + '_'):
                key = path.stem[len(prefix) + 1:]
                if key:
                    images[prefix].setdefault(key, []).append(path)
    ambiguous = False
    for prefix, mapping in images.items():
        for key, paths in sorted(mapping.items()):
            if len(paths) > 1:
                ambiguous = True
                warnings.append(f'{prefix}_{key} 同编号有多个文件，请保留明确来源后再导入：' +
                                '、'.join(p.name for p in paths))
            if key not in images['aux' if prefix == 'rgb_left' else 'rgb_left']:
                absent = 'aux' if prefix == 'rgb_left' else 'rgb_left'
                missing.append(f'{Path("multimodal") / paths[0].name} 缺少对应 {absent}_{key}.*')
    keys = sorted(set(images['rgb_left']) & set(images['aux']))
    if keys:
        ready.append(f'RGB+IR/辅助配对原图：{len(keys)} 组；需后台检测和逐对复核')
    else:
        warnings.append('尚无完整融合配对原图，可连接 RGB 左目和红外后继续采集')
    if not session.metadata.get('auxiliary_camera'):
        warnings.append('缺少辅助相机配置：请配置红外地址、模态及独立标定板')
    lines = ['会话文件检查', *ready, *('缺少：' + s for s in missing),
             *('提示：' + s for s in warnings),
             'OpenCV/Kalibr 等导出格式不是恢复双目结果的必需文件；未生成不会阻止续采。']
    return dict(aux_pairs=len(keys), auto_import=bool(keys) and not ambiguous and
                bool(session.metadata.get('auxiliary_camera')), missing=missing,
                warnings=warnings, text='\n'.join(lines))


def prepare_session_restore(session, *, allow_uncomputed=False):
    rig = session.load_rig()
    rig = (validate_session_rig(session, allow_legacy_offline=True) if rig is not None
           else MultiCameraRig() if allow_uncomputed else validate_session_rig(session))
    metadata = session.metadata
    board = CharucoBoard.from_dict(metadata['board'])
    model = CameraModel(metadata.get('camera_model', 'pinhole'))
    recorded = {pair['name']: pair for pair in metadata.get('pairs', [])}
    bindings = [(name, pair.left_id, pair.right_id) for name, pair in rig.pairs.items()]
    if not bindings:
        bindings = [(name, p['left_id'], p['right_id']) for name, p in recorded.items()]
    if not bindings:
        raise ValueError('会话缺少相机组绑定，无法恢复原图采集组')
    pairs = [StereoPairConfig(name,
        CameraConfig(left, '', 'left', name, recorded.get(name, {}).get('stream_type', 'rgb')),
        CameraConfig(right, '', 'right', name, recorded.get(name, {}).get('stream_type', 'rgb')))
        for name, left, right in bindings]
    camera_ids = [camera.camera_id for pair in pairs for camera in (pair.left, pair.right)]
    if (any(not isinstance(cid, str) or not cid.strip() for cid in camera_ids) or
            len(set(camera_ids)) != len(camera_ids)):
        raise ValueError('会话相机 ID 为空或重复，不能建立离线相机组')
    aux = metadata.get('auxiliary_camera')
    auxiliary = None
    if aux:
        auxiliary = AuxiliaryCameraConfig(
            camera_id=aux['camera_id'], url='', modality=aux.get('modality', 'ir'),
            board_pattern=aux.get('board_pattern', 'chessboard'),
            stream_type=aux.get('stream_type', 'ir'), group=aux.get('group', 'multimodal'),
            board_config=PatternBoardConfig.from_dict(aux['board_config'])
                         if aux.get('board_config') else None)
        if (not isinstance(auxiliary.camera_id, str) or not auxiliary.camera_id.strip() or
                auxiliary.camera_id in camera_ids):
            raise ValueError('辅助相机 ID 为空或与 RGB 相机重复')
        if auxiliary.modality not in ('ir', 'acoustic_rgb'):
            raise ValueError('会话辅助相机模态无效')
        auxiliary.pattern_board_config()
    return SessionRestorePlan(rig, board, model, pairs, auxiliary,
        any(is_legacy_offline_pair(name, pair, recorded) for name, pair in rig.pairs.items()))
