"""Preview entry points must not keep loading old disabled-control scripts."""
import hashlib

from scripts.jetson.web_assets import versioned_index


def test_content_change_replaces_script_url_and_entry_revision(tmp_path):
    (tmp_path / 'index.html').write_text(
        '<head><script src="/human.js" defer></script>'
        '<link href="/human.css" rel="stylesheet"></head>', encoding='utf-8')
    (tmp_path / 'human.js').write_text('old disabled controls')
    (tmp_path / 'human.css').write_text('body {}')
    first, old_revision = versioned_index(tmp_path)
    old_digest = hashlib.sha256(b'old disabled controls').hexdigest()[:16]
    assert f'/human.js?v={old_digest}' in first
    (tmp_path / 'human.js').write_text('enabled live controls')
    second, new_revision = versioned_index(tmp_path)
    assert f'/human.js?v={old_digest}' not in second
    new_digest = hashlib.sha256(b'enabled live controls').hexdigest()[:16]
    assert f'/human.js?v={new_digest}' in second
    assert new_revision != old_revision
    assert versioned_index(tmp_path) == (second, new_revision)


def test_asset_versioning_preserves_external_urls_and_page_controls(tmp_path):
    source = ('<head><script src="https://example.com/external.js"></script></head>'
              '<input id="human-show-trails" type="checkbox" checked>'
              '<input id="human-show-fov" type="checkbox" checked>')
    (tmp_path / 'index.html').write_text(source, encoding='utf-8')
    rendered, _ = versioned_index(tmp_path)
    assert 'src="https://example.com/external.js"' in rendered
    assert rendered.count('checked') == 2
    assert 'disabled' not in rendered
