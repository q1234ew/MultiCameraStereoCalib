"""Validate approved icon resources without constructing application windows."""

from pathlib import Path
import runpy
import struct

import pytest
from PySide6.QtGui import QIcon, QImage

from multicamera.runtime_paths import logo_png_path


ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / 'scripts/build_app_icon.py'))


def check_ico(path):
    data = path.read_bytes()
    assert struct.unpack_from('<HHH', data) == (0, 1, 7)
    sizes = []
    for index in range(7):
        width, height, _, _, planes, depth, length, offset = struct.unpack_from(
            '<BBBBHHII', data, 6 + index * 16)
        width, height = width or 256, height or 256
        assert planes == 1 and depth == 32
        assert offset >= 118 and offset + length <= len(data)
        image = QImage.fromData(data[offset:offset + length], 'PNG')
        assert not image.isNull() and (image.width(), image.height()) == (width, height)
        assert image.depth() == 32
        sizes.append(width)
    assert tuple(sizes) == BUILD['ICON_SIZES']


def test_master_and_ico_are_readable(app):
    assert logo_png_path().resolve() == (ROOT / 'assets/logo.png').resolve()
    image = QImage(str(logo_png_path()))
    assert image.width() == image.height() == 1254
    assert (ROOT / 'assets/logo-legacy.png').exists()
    check_ico(ROOT / 'assets/logo.ico')
    icon = QIcon(str(ROOT / 'assets/logo.ico'))
    for size in BUILD['ICON_SIZES']:
        assert not icon.pixmap(size, size).isNull()


def test_icon_can_be_regenerated_and_invalid_input_does_not_replace_it(tmp_path):
    destination = tmp_path / 'logo.ico'
    BUILD['build_icon'](ROOT / 'assets/logo.png', destination)
    check_ico(destination)
    before = destination.read_bytes()
    with pytest.raises(ValueError):
        BUILD['build_icon'](tmp_path / 'missing.png', destination)
    assert destination.read_bytes() == before
