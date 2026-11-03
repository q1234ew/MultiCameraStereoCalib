"""Read-only preparation of a historical calibration's offline workspace."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path

from ..calibration.aux_binding import auxiliary_binding_issue
from ..board.charuco_board import CharucoBoard
from ..calibration.models import CameraModel, MultiCameraRig
from ..calibration.offline_aux import file_identity, is_legacy_offline_pair, validate_session_rig
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
    identities: dict

    def verify_sources(self):
        _verify_sources(self.identities)


def _verify_sources(identities):
    for path, expected in identities.items():
        actual = file_identity(path)['sha256'] if path.is_file() else None
        if actual != expected:
            raise ValueError(f'会话文件在准备期间变化，未切换工作区：{path}')


def _source_identities(session):
    # Capture the pointer before resolving it; recheck both pointer and target after parsing.
    session.verify_loaded_metadata()
    root = session.session_dir
    paths = [root / session.METADATA_FILE, root / session.RESULTS_DIR / 'latest.json']
    identities = {p: file_identity(p)['sha256'] if p.is_file() else None for p in paths}
    path = session.result_path
    identities[path] = file_identity(path)['sha256'] if path.is_file() else None
    return identities


@dataclass
class PreparedSessionResult:
    rig: MultiCameraRig
    historical_aux: dict
    directory: Path
    files: list[Path]
    errors: list[str]
    provenance: dict
    aux_id: str | None
    aux_model_key: str | None
    aux_status: str
    resume_state: dict
    identities: dict

    def verify_sources(self):
        _verify_sources(self.identities)


def prepare_session_result(session, rig, auxiliary=None):
    """All file reads and declared AUX validation precede any workspace mutation."""
    from ..calibration.joint_aux import AUX_MODELS
    from ..calibration.quality_summary import aux_metrics_text, summarize_quality
    from .export import _validate_rig

    session.verify_loaded_metadata()
    directory = session.result_path.parent
    manifest = directory / 'calibration_manifest.json'
    sources = (session.result_path, session.session_dir / session.METADATA_FILE,
               session.session_dir / session.RESULTS_DIR / 'latest.json', manifest)
    identities = {path: file_identity(path)['sha256'] if path.is_file() else None for path in sources}
    loaded = session.load_rig()
    def canonical(value):
        return json.loads(json.dumps(value.to_dict(), allow_nan=False))
    if loaded is None or canonical(loaded) != canonical(rig):
        raise ValueError('会话结果与准备的参数不一致，未切换工作区')
    for field in ('frame_count', 'multimodal_frame_count'):
        value = session.metadata.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'会话 {field} 必须是非负整数')
    active = deepcopy(rig)
    historical = {}
    rgb_ids = {cid for pair in active.pairs.values() for cid in (pair.left_id, pair.right_id)}
    # Validate malformed declarations even if the AUX ID/config itself mismatches.
    for cid, aux in list(active.aux_cameras.items()):
        reason = auxiliary_binding_issue(active, aux)
        if auxiliary is None or cid != auxiliary.camera_id:
            reason = reason or '会话未配置该辅助相机'
        elif aux.modality != auxiliary.modality:
            reason = reason or '辅助模态与会话不一致'
        elif aux.board_config is not None:
            board = PatternBoardConfig.from_dict(aux.board_config)
            if board != auxiliary.pattern_board_config():
                reason = reason or '辅助板配置与会话不一致'
        aux_metrics_text(aux)  # Reject unrenderable legacy/result metadata before committing.
        if reason:
            historical[cid] = {'calibration': aux, 'reason': reason}
            del active.aux_cameras[cid]
            if cid not in rgb_ids:
                active.extrinsics.pop(cid, None)
    _validate_rig(active)
    for pair in active.pairs.values():
        for value in (pair.left_intrinsics.rms_error, pair.right_intrinsics.rms_error,
                      pair.stereo.rms_error):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError('会话RMS必须是有限非负数值')
    summarize_quality(active)
    provenance = dict(kind='restored_session_result', result=file_identity(session.result_path),
                      session_name=session.name)
    files = ([session.result_path] if directory == session.session_dir else
             sorted(path for path in directory.glob('*') if path.is_file()))
    errors = []
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not isinstance(data.get('errors', []), list) or any(
                not isinstance(value, str) for value in data.get('errors', [])):
            raise ValueError('calibration_manifest.json 的 errors 必须是字符串列表')
        errors = data.get('errors', [])
    resume = session.multimodal_resume_state()
    session.next_stereo_frame_index
    aux_id = auxiliary.camera_id if auxiliary and auxiliary.camera_id in active.aux_cameras else None
    model_key, status = None, ''
    if aux_id:
        aux = active.aux_cameras[aux_id]
        key = aux.calibration_info.get('aux_model')
        if isinstance(key, str) and key in AUX_MODELS and AUX_MODELS[key][1] == aux.intrinsics.model:
            model_key = key
        status = aux_metrics_text(aux)
    result = PreparedSessionResult(active, historical, directory, files, errors, provenance,
                                   aux_id, model_key, status, resume, identities)
    result.verify_sources()
    return result


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
    identities = _source_identities(session)
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
    session.verify_loaded_metadata()
    _verify_sources(identities)
    return SessionRestorePlan(rig, board, model, pairs, auxiliary,
        any(is_legacy_offline_pair(name, pair, recorded) for name, pair in rig.pairs.items()),
        identities)
