"""Keep one QApplication alive until all widgets and workers have been cleaned up."""

import gc
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import cv2
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication


def pytest_configure(config):
    value = os.environ.get('MULTICAMERA_TEST_OPENCV_THREADS')
    if value is None:
        return
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise pytest.UsageError('MULTICAMERA_TEST_OPENCV_THREADS must be a positive integer')
    # GCD only supports nonpositive values; zero disables OpenCV parallel regions.
    gcd = any(line.split(':', 1)[-1].strip() == 'GCD'
              for line in cv2.getBuildInformation().splitlines()
              if 'Parallel framework:' in line)
    cv2.setNumThreads(0 if gcd else int(value))


@pytest.fixture
def finish_aux_tasks(app):
    def finish(panel):
        deadline = time.monotonic() + 15
        while (panel._aux_capture_worker or panel._aux_guide._worker
                or panel._aux_guide._progress_worker or panel._aux_import_worker):
            app.processEvents()
            if time.monotonic() >= deadline:
                pytest.fail('Auxiliary capture/guidance did not finish')
            time.sleep(.002)
        app.processEvents()
    return finish


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    application._style_reference = application.style()
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    application.processEvents()
    gc.collect()


@pytest.fixture(autouse=True)
def cleanup_test_widgets(request):
    if "app" not in request.fixturenames:
        yield
        return
    application = request.getfixturevalue("app")
    existing = {id(widget) for widget in application.topLevelWidgets()}
    yield
    # Closed Qt windows remain alive unless explicitly deleted, including their timers.
    for widget in application.topLevelWidgets():
        if id(widget) not in existing:
            widget.close()
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    application.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
