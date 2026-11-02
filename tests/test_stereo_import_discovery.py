"""Recursive imports keep unambiguous source identity without renumbering."""

import cv2
import numpy as np
import pytest
from PySide6.QtTest import QSignalSpy

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.ui.threads.import_worker import StereoImageImportWorker, _discover_stereo_pairs


def create_pair(directory, key, extension='.png'):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f'{key}_{side}{extension}' for side in ('left', 'right')]
    for path in paths:
        path.touch()
    return paths


def test_duplicate_basenames_get_relative_file_identity(tmp_path):
    create_pair(tmp_path / 'sessionA', '0001')
    create_pair(tmp_path / 'sessionB', '0001')
    create_pair(tmp_path, '0011')
    pairs = _discover_stereo_pairs(str(tmp_path))
    assert [key for key, *_ in pairs] == ['0011', 'sessionA/0001_left.png', 'sessionB/0001_left.png']
    assert len(set(key for key, *_ in pairs)) == 3


def test_same_stem_different_formats_stays_distinguishable(tmp_path):
    create_pair(tmp_path, '0001', '.png')
    create_pair(tmp_path, '0001', '.jpg')
    assert {key for key, *_ in _discover_stereo_pairs(str(tmp_path))} == {'0001_left.png', '0001_left.jpg'}


def test_simple_source_ids_and_gaps_unchanged(tmp_path):
    create_pair(tmp_path, '0011')
    create_pair(tmp_path, '0008')
    assert [key for key, *_ in _discover_stereo_pairs(str(tmp_path))] == ['0008', '0011']


def test_case_colliding_paths_refused_on_case_sensitive_filesystem(tmp_path):
    create_pair(tmp_path, 'Case')
    create_pair(tmp_path, 'case')
    if len(list(tmp_path.iterdir())) != 4:
        pytest.skip('Filesystem cannot create distinct case-only paths')
    with pytest.raises(ValueError, match='大小写'):
        _discover_stereo_pairs(str(tmp_path))


def test_unpaired_image_is_reported_without_misleading_pair_count(app, tmp_path):
    image = np.zeros((48, 64, 3), np.uint8)
    paths = create_pair(tmp_path, '0001')
    for path in [*paths, tmp_path / '0002_left.png']:
        assert cv2.imwrite(str(path), image)
    worker = StereoImageImportWorker(str(tmp_path), CharucoBoard(CharucoBoardConfig()))
    logs, results, failures = QSignalSpy(worker.log), QSignalSpy(worker.finished_ok), QSignalSpy(worker.failed)
    worker.run()
    assert results.count() == 1 and failures.count() == 0
    assert len(results.at(0)[0]) == 1
    assert worker.source_paths == {'0001': tuple(str(path.resolve()) for path in paths)}
    assert any('1 张图片未参与配对' in logs.at(i)[0] for i in range(logs.count()))
