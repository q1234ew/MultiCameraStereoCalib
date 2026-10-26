"""Opt-in sustained Qt calibration workflow QA in isolated temporary sessions."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'data/pairs')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=10)
    parser.add_argument('--duration-seconds', type=float, default=0.)
    parser.add_argument('--dwell-seconds', type=float, default=2.)
    parser.add_argument('--platform', choices=('cocoa', 'offscreen', 'windows'), default='offscreen')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.cycles < 1 or args.dwell_seconds < 0 or args.duration_seconds < 0:
        parser.error('Cycle count must be positive; durations must be nonnegative')
    os.environ['QT_QPA_PLATFORM'] = args.platform
    args.report.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(args.report.with_suffix('.log')), level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s', encoding='utf-8')
    fault_log = args.report.with_suffix('.fault.log').open('wb', buffering=0)
    faulthandler.enable(file=fault_log, all_threads=True)

    import cv2
    from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication

    from multicamera.calibration.models import CameraModel
    from multicamera.ui import main_window
    from multicamera.ui.theme import configure_application_style
    from multicamera.ui.threads.import_worker import StereoImageImportWorker, _discover_stereo_pairs

    source = args.source.resolve()
    code_root = Path(main_window.__file__).resolve().parents[1]

    def code_hashes():
        return {str(path.relative_to(code_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in code_root.rglob('*.py')}
    paths = sorted({path for _, left, right in _discover_stereo_pairs(str(source)) for path in (left, right)})
    if not paths:
        parser.error('No source stereo pairs found')
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    configure_application_style(app)
    started = time.monotonic()
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'platform': args.platform, 'opencv': cv2.__version__, 'source': str(source),
              'input_hashes': input_hashes, 'cycles': [],
              'source_hashes_start': code_hashes(),
              'limitations': 'Local Qt process only; not a Windows EXE or real-device acceptance test.'}

    def persist():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        staged = args.report.with_suffix('.tmp')
        staged.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        staged.replace(args.report)

    def wait_for(predicate, timeout=120.):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError('Calibration workflow did not become idle')
            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec()
        app.processEvents()

    persist()
    try:
        with TemporaryDirectory(prefix='multicamera-workflow-soak-') as temp:
            main_window.sessions_dir = lambda: Path(temp) / 'sessions'
            for index in range(args.cycles):
                if index and args.duration_seconds and time.monotonic() - started >= args.duration_seconds:
                    break
                cycle_started = time.monotonic()
                record = {'index': index + 1, 'status': 'running', 'phase': 'construct_window'}
                report['cycles'].append(record)
                persist()
                win = main_window.MainWindow()
                panel = win._calib_panel
                try:
                    width, height = ((1000, 680), (1280, 800), (1440, 1024))[index % 3]
                    win.resize(width, height)
                    if args.show:
                        win.show()
                    model = CameraModel.FISHEYE if index % 2 else CameraModel.PINHOLE
                    win._camera_model = model
                    record['phase'] = 'model_configuration'
                    persist()
                    panel.set_camera_model(model)
                    panel._pending_import_folder = source
                    worker = StereoImageImportWorker(str(source), panel._board, parent=panel)
                    worker.log.connect(panel.append_log)
                    worker.finished_ok.connect(panel._on_import_images_ready)
                    worker.failed.connect(panel._on_import_images_failed)
                    worker.finished.connect(worker.deleteLater)
                    panel._stereo_import_worker = worker
                    panel._set_import_busy(True)
                    worker.start()
                    record['phase'] = 'import'
                    persist()
                    wait_for(lambda: not panel.has_active_background_tasks())
                    assert panel._stereo_cals['offline'].ready
                    policy = 'fisheye_common' if model == CameraModel.FISHEYE else ('quality' if index % 4 else 'all')
                    panel._combo_data_filter.setCurrentIndex(panel._combo_data_filter.findData(policy))
                    strategy = 'joint_refine' if index % 4 == 3 else 'standard'
                    panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData(strategy))
                    panel._calibrate_pair('offline')
                    record['phase'] = 'calibration'
                    persist()
                    closing = QCloseEvent()
                    win.closeEvent(closing)
                    assert not closing.isAccepted(), 'Window closed while calibration was pending'
                    wait_for(lambda: not panel.has_active_background_tasks())
                    assert win._rig is not None, panel._log_text.toPlainText()
                    pair = win._rig.pairs['offline']
                    assert pair.left_intrinsics.frame_ids
                    panel._show_frame_diagnostics()
                    record['phase'] = 'diagnostics'
                    persist()
                    dialog = panel._diagnostic_dialog
                    dialog._table.setCurrentCell(0, 0)
                    dialog._btn_preview.click()
                    assert win._stream_view.review_key, 'Diagnostics failed to locate original image'
                    dialog.close()
                    win._stream_view.end_review()
                    manifest_path = win._current_session.session_dir / 'results/calibration_manifest.json'
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    for item in manifest['files']:
                        data = (manifest_path.parent / item['name']).read_bytes()
                        assert hashlib.sha256(data).hexdigest() == item['sha256']
                    record.update(model=model.value, policy=policy, strategy=strategy,
                                  size=[width, height], frames=panel._stereo_cals['offline'].num_frames,
                                  stereo_rms=pair.stereo.rms_error,
                                  decision=panel._lbl_result_decision.text(),
                                  save_errors=manifest['errors'])
                    loop = QEventLoop()
                    record['phase'] = 'dwell'
                    persist()
                    pages = ('capture', 'calibration', 'results')
                    tick = [0]
                    page_timer = QTimer()

                    def navigate():
                        panel.show_workflow(pages[tick[0] % len(pages)])
                        tick[0] += 1

                    page_timer.timeout.connect(navigate)
                    page_timer.start(1000)
                    QTimer.singleShot(int(args.dwell_seconds * 1000), loop.quit)
                    loop.exec()
                    page_timer.stop()
                    record.update(status='passed', navigation_ticks=tick[0],
                                  phase='closing',
                                  elapsed_seconds=round(time.monotonic() - cycle_started, 3))
                finally:
                    wait_for(lambda: not panel.has_active_background_tasks())
                    win._stream_manager.stop_all()
                    assert win.close()
                    win.deleteLater()
                    panel = win = None
                    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                    app.processEvents()
                    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                    gc.collect()
                record['remaining_visible_windows'] = sum(widget.isVisible() for widget in app.topLevelWidgets())
                assert record['remaining_visible_windows'] == 0
                record['phase'] = 'complete'
                persist()
                print(f"Cycle {index + 1}: {record['model']} / {record['strategy']} passed", flush=True)
        report['input_unchanged'] = input_hashes == {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        }
        assert report['input_unchanged']
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report['source_hashes_end'] = code_hashes()
        report['source_changed_during_run'] = report['source_hashes_start'] != report['source_hashes_end']
        persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
