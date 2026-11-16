"""Content-versioned entry page for long-lived camera-preview browser sessions."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def versioned_index(assets: Path):
    """Give local scripts/styles new URLs whenever their actual contents change."""
    source = (assets / 'index.html').read_text(encoding='utf-8')

    def replace(match):
        name = match.group(2)
        digest = hashlib.sha256((assets / name).read_bytes()).hexdigest()[:16]
        return f'{match.group(1)}/{name}?v={digest}"'

    source = re.sub(r'((?:src|href)=")/([\w-]+\.(?:js|css))"', replace, source)
    revision = hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]
    source = source.replace('</head>',
                            f'<meta name="multicamera-ui-revision" content="{revision}"></head>')
    return source, revision
