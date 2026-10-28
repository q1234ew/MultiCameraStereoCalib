"""Admission evidence survives failed undo; source integrity covers real naming styles."""

import json
from pathlib import Path

import pytest

from multicamera.io.session import CalibrationSession
from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.ui.threads.import_worker import StereoImageImportWorker
from test_real_pairs_ui import source_image_hashes


@pytest.mark.parametrize('locked_role', ['left', 'right'])
def test_failed_image_delete_retains_quality_until_retry(tmp_path, monkeypatch, locked_role):
    session = CalibrationSession(tmp_path)
    folder = tmp_path / 'images/pair'
    folder.mkdir()
    paths = {role: folder / f'{role}_0001.png' for role in ('left', 'right')}
    for path in paths.values():
        path.write_bytes(b'original')
    session.save_capture_review('pair', 1, {'state': '拒收', 'reasons': ['test']})
    quality = folder / 'quality_0001.json'
    unlink = Path.unlink

    def locked_unlink(path, *args, **kwargs):
        if path == paths[locked_role]:
            raise PermissionError('simulated Windows image lock')
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', locked_unlink)
        with pytest.raises(PermissionError):
            session.delete_saved_pair_frame('pair', 1)
    assert quality.exists()
    assert json.loads(quality.read_text(encoding='utf-8'))['state'] == '拒收'
    assert paths[locked_role].exists()
    worker = StereoImageImportWorker(str(folder), CharucoBoard(CharucoBoardConfig()))
    failures = []
    worker.failed.connect(failures.append)
    with monkeypatch.context() as patch:
        patch.setattr('multicamera.ui.threads.import_worker.cv2.imread',
                      lambda *args: pytest.fail('Rejected or incomplete pair was read'))
        worker.run()
    assert failures
    assert session.delete_saved_pair_frame('pair', 1)
    assert not quality.exists() and not any(path.exists() for path in paths.values())


@pytest.mark.parametrize('names', [
    ('left_0001.png', 'right_0001.png'),
    ('0001_left.png', '0001_right.png'),
    ('left/0001.png', 'right/0001.png'),
])
def test_source_hashes_include_both_roles_and_detect_changes(tmp_path, names):
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unchanged')
    before = source_image_hashes(tmp_path)
    assert len(before) == 2
    (tmp_path / names[1]).write_bytes(b'changed')
    assert before != source_image_hashes(tmp_path)


def test_empty_source_hash_check_cannot_pass_vacuously(tmp_path):
    with pytest.raises(AssertionError, match='No paired source'):
        source_image_hashes(tmp_path)
