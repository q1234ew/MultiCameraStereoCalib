"""Convert the approved square PNG to a multi-resolution Windows ICO."""

import argparse
from pathlib import Path
import struct

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PySide6.QtGui import QImage


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def build_icon(source: Path, destination: Path):
    image = QImage(str(source))
    if image.isNull() or image.width() != image.height():
        raise ValueError('Icon source must be a readable square image')
    entries, payloads = [], []
    offset = 6 + 16 * len(ICON_SIZES)
    for size in ICON_SIZES:
        scaled = image.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation).convertToFormat(
            QImage.Format_ARGB32)
        data = QByteArray()
        buffer = QBuffer(data)
        if not buffer.open(QIODevice.WriteOnly) or not scaled.save(buffer, 'PNG'):
            raise RuntimeError(f'Unable to encode {size}px icon')
        buffer.close()
        payload = bytes(data)
        entries.append(struct.pack('<BBBBHHII', size % 256, size % 256, 0, 0,
                                   1, 32, len(payload), offset))
        payloads.append(payload)
        offset += len(payload)
    destination.write_bytes(struct.pack('<HHH', 0, 1, len(entries)) +
                            b''.join(entries) + b''.join(payloads))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root / 'assets/logo.png')
    parser.add_argument('--output', type=Path, default=root / 'assets/logo.ico')
    args = parser.parse_args()
    build_icon(args.source, args.output)
    print(f'Created {args.output} ({len(ICON_SIZES)} sizes)')
