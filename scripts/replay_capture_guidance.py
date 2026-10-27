"""Replay numbered offline stereo images without using future observations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import CharucoDetector
from multicamera.calibration.adaptive_guidance import GuidanceObservation
from multicamera.calibration.guidance_replay import replay_observations
from multicamera.calibration.models import CameraModel
from multicamera.ui.threads.import_worker import _discover_stereo_pairs


def run(source, output, board, model):
    source, output = Path(source).resolve(), Path(output).resolve()
    pairs = _discover_stereo_pairs(str(source))
    if not pairs:
        raise ValueError('未找到配对图像')
    output.mkdir(parents=True, exist_ok=False)
    detector = CharucoDetector(board)
    sources = {}

    def frames():
        for frame_id, left, right in pairs:
            observations, files = {}, {}
            for camera, path in (('left', left), ('right', right)):
                data = path.read_bytes()
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                files[camera] = {'path': str(path.resolve()), 'url': path.resolve().as_uri(),
                                 'sha256': hashlib.sha256(data).hexdigest(), 'readable': image is not None}
                detection = detector.detect(image) if image is not None else None
                observations[camera] = GuidanceObservation(
                    detection.charuco_corners, detection.charuco_ids, detection.image_size, frame_id,
                ) if detection is not None and detection.valid else None
                files[camera]['detected_corners'] = detection.num_corners if detection is not None else 0
            sources[frame_id] = files
            print(f'回放 {frame_id}: L={files["left"]["detected_corners"]} '
                  f'R={files["right"]["detected_corners"]}', flush=True)
            yield frame_id, observations

    report = replay_observations(board, model, frames())
    for row in report['steps']:
        row['files'] = sources[row['frame_id']]
    for files in sources.values():
        for metadata in files.values():
            if hashlib.sha256(Path(metadata['path']).read_bytes()).hexdigest() != metadata['sha256']:
                raise RuntimeError('回放期间源图片发生变化，不能交付本次报告')
    report.update({'source': str(source), 'board': board.to_dict(), 'model': model.value,
                   'source_hashes_verified': True, 'order': 'natural_filename_order'})
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (output / 'report.json').write_text(encoded, encoding='utf-8')
    template = Path(__file__).with_name('templates') / 'guidance_replay.html'
    (output / 'index.html').write_text(template.read_text(encoding='utf-8').replace(
        '__REPLAY_DATA__', encoded.replace('<', '\\u003c')), encoding='utf-8')
    lines = ['# 双目采图引导离线回放', '', f'输入 {len(pairs)} 对 · 模型 {model.value}', '',
             report['final']['summary'], '', '## 最终建议', '',
             report['final']['recommendation']['reason'], '', '## 逐帧对照', '',
             '| 图片 | 此前可用引导内参 | 去重样本变化 | 对此前建议的匹配 |', '| --- | --- | --- | --- |']
    lines.extend(f'| {r["frame_id"]} | {len(r["before"]["intrinsic_cameras"])}/2 | '
                 f'{r["unique_sample_change"]:+d} | {r["verdict"]["reason"]} |' for r in report['steps'])
    lines.extend(['', '## 边界', '', *['- ' + limit for limit in report['limits']],
                  '- 所有原图哈希复查通过；报告不写入会话或覆盖标定结果。',
                  '- HTML 引用本机原图，不复制图片；移动报告到其他电脑需要保持原图路径可访问。',
                  '- 无板检测、几何不合格和重复构图不会被默认为已完成采图目标。'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def main():
    # Windows redirected streams can otherwise inherit an ANSI code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/pairs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--board-config', type=Path, help='ChArUco board JSON; default matches data/pairs')
    parser.add_argument('--model', choices=[m.value for m in CameraModel], default='pinhole')
    args = parser.parse_args()
    board = (CharucoBoard.from_dict(json.loads(args.board_config.read_text(encoding='utf-8')))
             if args.board_config else CharucoBoard(CharucoBoardConfig()))
    result = run(args.input, args.output, board, CameraModel(args.model))
    print(result['final']['summary'])
    print(f'报告：{args.output.resolve() / "index.html"}')


if __name__ == '__main__':
    main()
