from html.parser import HTMLParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "frontend" / "static" / "ancserAPX.html"
CSS_PATH = ROOT / "frontend" / "static" / "ancserTPX.css"
JS_PATH = ROOT / "frontend" / "static" / "ancserAPX.js"
GLASS_JS_PATH = ROOT / "frontend" / "static" / "apx-liquid-glass.js"
DOC_PATH = ROOT / "docs" / "APX_GLASS_STANDARD.md"

sys.path.insert(0, str(ROOT))
from scripts.sync_glass_shader import build_production_shaders  # noqa: E402


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
    assert "label.className = 'factor-chk apx-factor-switch apx-glass-switch'" in js
    assert "APX Liquid Glass Standard" in doc
    assert "Preserve IDs, values, and event handlers" in doc


def test_light_mode_switch_is_wired_end_to_end():
    html = HTML_PATH.read_text(encoding="utf-8")
    css = CSS_PATH.read_text(encoding="utf-8")
    js = JS_PATH.read_text(encoding="utf-8")
    module = GLASS_JS_PATH.read_text(encoding="utf-8")

    # the switch itself is a glass switch, mounted by the same adapter
    assert 'id="theme-toggle"' in html
    assert "apx-theme-switch apx-glass-switch" in html
    assert ':root[data-theme="light"]' in css
    assert "function setTheme" in js
    assert "function chartThemeOptions" in js
    # the renderer repaints its backdrop and scene fills for the new palette
    assert "dataset.theme === 'light'" in module
    assert "BACKDROP" in module and "SCENE" in module


def test_webgl_glass_renderer_is_loaded_and_shares_one_context():
    html = HTML_PATH.read_text(encoding="utf-8")
    module = GLASS_JS_PATH.read_text(encoding="utf-8")

    assert "/static/apx-liquid-glass.js" in html
    # One shared WebGL2 context for the whole page: a context per control would
    # blow past the browser cap with ~25 glass controls on screen.
    assert module.count('getContext(\'webgl2\'') == 1
    assert "function initRenderer" in module
    assert "state.glCanvas" in module
    # Idle controls must keep their last frame instead of re-rendering.
    assert "state.raf = running ? requestAnimationFrame(tick) : 0;" in module
    # Fallback path stays intact when WebGL is unavailable.
    assert "state.disabled = true" in module
    assert "dataset.apxGlassRenderer = 'css'" in module


def test_webgl_glass_mounts_every_standardized_control():
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    for selector in (
        ".apx-glass-nav",
        ".apx-glass-switch input[type=checkbox]",
        "input[type=range].apx-glass-range",
        ".dual-range",
        ".btn.apx-glass-action",
        ".toggle-field.apx-glass-toggle",
        ".conn-trigger.apx-glass-status",
        ".account-badge.apx-glass-badge",
    ):
        assert selector in module, selector


def test_switches_and_sliders_rest_flat_like_the_demo_bench():
    """Idle means no glass: a solid grey thumb, as in drawBarScene()."""
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    assert "function quietWeight" in module
    assert "paintQuietThumb" in module
    # the spring from updateLiquidToggle(), overshoot included
    assert "springMotion" in module
    assert "1.16" in module
    # only surfaces keep a resting rim
    assert module.count("idleInteraction: 0,") == 4
    assert module.count("idleInteraction: 0.34,") == 1


def test_controls_return_to_idle_on_release():
    """A click leaves DOM focus behind; that must not pin the lens open."""
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    assert "hoverActivates" in module
    assert "node.focused = state.keyboardNav;" in module
    assert "keyboardNav: false," in module


def test_switch_uses_the_demo_layering():
    """The bench keeps track and thumb in the DOM and cross-fades the canvas
    over them, so no CSS rule can ever cover the rendered surface."""
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    css = CSS_PATH.read_text(encoding="utf-8")
    toggle = module.split("function mountToggle", 1)[1].split("    // renderLiquid", 1)[0]
    # canvas paints the lens only — the track is a DOM element
    assert "trackVisibility: 0," in toggle
    assert "apx-switch-track" in toggle and "apx-switch-thumb" in toggle
    assert "function syncToggleDom" in module
    # cross-fade: canvas fades in exactly as the solid thumb fades out
    assert "opacity: var(--apx-switch-glass, 0);" in css
    assert "opacity: calc(1 - var(--apx-switch-glass, 0));" in css
    # …and the scene must take the thumb over, or holding the switch refracts a
    # flat track and the ball vanishes
    paint = module.split("function paintTogglePillSurface", 1)[1].split("\n    }", 1)[0]
    assert "const glass = Math.min(1, node.activity);" in paint
    assert "solidThumbColor(progress, glass" in paint


def test_css_always_yields_where_the_canvas_owns_the_surface():
    """A theme rule outranks a plain hand-off rule and its fill then covers the
    canvas — light-mode OFF switches went invisible exactly this way."""
    css = CSS_PATH.read_text(encoding="utf-8")
    owned = (
        '.apx-glass-switch input[type="checkbox"][data-apx-renderer="webgl"],',
        'input[type="range"].apx-glass-range[data-apx-renderer="webgl"] {',
        '.panel.apx-glass-container[data-apx-renderer="webgl"] {',
        '.account-badge.apx-glass-badge[data-apx-renderer="webgl"] {',
    )
    for selector in owned:
        block = css.split(selector, 1)[1].split("}", 1)[0]
        assert "background: none !important" in block, selector


def test_switch_thumb_is_a_capsule_with_stronger_centre_shrink():
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    aspect = float(
        module.split("const TOGGLE_THUMB_ASPECT = ", 1)[1].split(";", 1)[0]
    )
    assert aspect > 1, "a square thumb renders as a circle, not a capsule"
    toggle = module.split("function mountToggle", 1)[1].split("function ", 1)[0]
    assert "tuning: { centerShrink:" in toggle
    # track colours are painted by the canvas, so their contrast lives in SCENE
    assert "pillOn:" in module


def test_swollen_thumb_refracts_track_not_the_page_behind_it():
    """Only the lens is masked outside the pill, so the track colour has to
    bleed past it or the swollen thumb shows the raw backdrop as a slab."""
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    assert "TRACK_BLEED_PX" in module
    assert "fillPill(-TRACK_BLEED_PX);" in module


def test_lens_size_eases_between_targets():
    """Neighbouring nav tabs differ in width; the lens must grow into the new
    one instead of snapping."""
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    assert "function syncLensSizes" in module
    assert "function easeLensSizes" in module
    assert "node.sizes[index] || lens" in module


def test_slider_grips_are_wide_capsules_like_the_bench():
    """The bench thumb is a wide capsule; a capsule SDF takes its radius from
    the short axis, so the grip must stay wider than it is tall."""
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    dual = module.split("function mountDualRange", 1)[1].split("function ", 1)[0]
    assert "shape: 'capsule'" in dual
    half_w = int(module.split("const DUAL_GRIP_HALF_W = ", 1)[1].split(";", 1)[0])
    half_h = int(module.split("const DUAL_GRIP_HALF_H = ", 1)[1].split(";", 1)[0])
    assert half_w > half_h, "an upright capsule collapses into a circle"


def test_panels_are_rendered_as_real_glass_containers():
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    css = CSS_PATH.read_text(encoding="utf-8")
    assert "selector: '.panel.apx-glass-container'," in module
    # the CSS fill must step aside so the canvas is the visible surface
    assert '.panel.apx-glass-container[data-apx-renderer="webgl"]' in css
    # the page gradient is the sidebar's background
    assert "background: transparent;" in css


def test_action_surfaces_do_not_deform_on_hover():
    module = GLASS_JS_PATH.read_text(encoding="utf-8")
    css = CSS_PATH.read_text(encoding="utf-8")
    assert "tuning: { centerShrink: 0, floatingShrink: 0 }" in module
    assert "grow: { x: 0, y: 0 }" in module
    assert "transform: none;" in css


def test_webgl_surfaces_replace_their_css_fallback_fill():
    css = CSS_PATH.read_text(encoding="utf-8")
    assert ".apx-glass-canvas" in css
    assert ".apx-glass-mount" in css
    for selector in (
        '.apx-glass-nav[data-apx-renderer="webgl"]',
        '.btn.apx-glass-action[data-apx-renderer="webgl"]',
        'input[type="range"].apx-glass-range[data-apx-renderer="webgl"]',
        '.apx-glass-switch input[type="checkbox"][data-apx-renderer="webgl"]::after',
    ):
        assert selector in css, selector


def test_production_shader_matches_the_demo_bench():
    """The demo is the calibration source; drift means the UI stops matching it.

    Regenerate with `python scripts/sync_glass_shader.py` after editing the demo.
    """
    vertex, fragment = build_production_shaders()
    module = GLASS_JS_PATH.read_text(encoding="utf-8")

    assert vertex in module, "vertex shader is out of sync with the demo bench"
    assert fragment in module, "fragment shader is out of sync with the demo bench"
    # The single documented deviation from the bench.
    assert "uniform float uCornerRadiusPx;" in fragment
    assert module.count("uCornerRadiusPx") >= 3
