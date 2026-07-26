from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "frontend" / "static" / "ancserAPX.html"
CSS_PATH = ROOT / "frontend" / "static" / "ancserTPX.css"
JS_PATH = ROOT / "frontend" / "static" / "ancserAPX.js"
DOC_PATH = ROOT / "docs" / "APX_GLASS_STANDARD.md"


class _ClassParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.by_id = {}
        self.glass_nodes = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.by_id[attributes["id"]] = attributes
        if attributes.get("data-apx-glass"):
            self.glass_nodes.append((tag, attributes))


def _classes(attrs):
    return set(attrs.get("class", "").split())


def test_apx_frontend_applies_glass_standard_to_requested_controls():
    source = HTML_PATH.read_text(encoding="utf-8")
    parser = _ClassParser()
    parser.feed(source)

    assert 'class="header-tabs apx-glass-nav"' in source
    assert 'class="bottom-tabs apx-glass-nav apx-glass-nav--bottom"' in source
    assert "apx-glass-status" in _classes(parser.by_id["conn-trigger"])
    assert "apx-glass-badge" in _classes(parser.by_id["account-badge"])
    assert "apx-glass-range" in _classes(parser.by_id["leverage-slider"])
    assert "apx-glass-action" in _classes(parser.by_id["btn-run-backtest"])
    assert "apx-glass-action--green" in _classes(parser.by_id["btn-run-backtest"])
    assert "apx-glass-action" in _classes(parser.by_id["btn-sweep-backtest"])
    assert "apx-glass-action" in _classes(parser.by_id["btn-apply-live"])
    assert "apx-glass-action--amber" in _classes(parser.by_id["btn-apply-live"])
    assert "apx-glass-action" in _classes(parser.by_id["btn-refresh-live"])
    assert any(attrs.get("data-apx-glass") == "nav" for _, attrs in parser.glass_nodes)
    assert any(attrs.get("data-apx-glass") == "switch-list" for _, attrs in parser.glass_nodes)
    assert any(attrs.get("data-apx-glass") == "range" for _, attrs in parser.glass_nodes)
    assert any(attrs.get("data-apx-glass") == "container" for _, attrs in parser.glass_nodes)


def test_apx_glass_standard_css_and_js_helpers_exist():
    css = CSS_PATH.read_text(encoding="utf-8")
    js = JS_PATH.read_text(encoding="utf-8")
    doc = DOC_PATH.read_text(encoding="utf-8")

    assert ".apx-glass-container" in css
    assert ".apx-glass-nav" in css
    assert ".factor-chk.apx-factor-switch" in css
    assert "input[type=\"range\"].apx-glass-range" in css
    assert ".btn.apx-glass-action" in css
    assert "backdrop-filter:" in css
    assert "function initApxGlassStandard" in js
    assert "function syncApxGlassRange" in js
    assert "syncApxFactorSwitches" in js
    assert "label.className = 'factor-chk apx-factor-switch'" in js
    assert "APX Liquid Glass Standard" in doc
    assert "preserve IDs, values, and event handlers" in doc
