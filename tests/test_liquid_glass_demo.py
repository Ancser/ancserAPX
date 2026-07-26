from html.parser import HTMLParser
from pathlib import Path


DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "static"
    / "demos"
    / "liquid-glass-switch.html"
)


class _DemoMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tablist_count = 0
        self.radiogroup_count = 0
        self.tabs = []
        self.radios = []
        self.canvases = []
        self.toggle_canvases = []
        self.color_canvases = []
        self.optical_switches = []
        self.switches = []
        self.ranges = []
        self.liquid_ranges = []
        self.motion_blocks = []
        self.external_scripts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("role") == "tablist":
            self.tablist_count += 1
        if attributes.get("role") == "radiogroup":
            self.radiogroup_count += 1
        if attributes.get("role") == "tab":
            self.tabs.append(attributes)
        if attributes.get("role") == "radio":
            self.radios.append(attributes)
        if attributes.get("role") == "switch":
            self.switches.append(attributes)
        if tag == "input" and attributes.get("type") == "range":
            self.ranges.append(attributes)
        if tag == "canvas":
            self.canvases.append(attributes)
            if "liquid-toggle-canvas" in attributes.get("class", "").split():
                self.toggle_canvases.append(attributes)
            if attributes.get("id") == "optical-color-canvas":
                self.color_canvases.append(attributes)
        if "data-optical-switch" in attributes:
            self.optical_switches.append(attributes)
        if "data-liquid-range" in attributes:
            self.liquid_ranges.append(attributes)
        if "data-motion-block" in attributes:
            self.motion_blocks.append(attributes)
        if tag == "script" and attributes.get("src"):
            self.external_scripts.append(attributes["src"])


def test_liquid_glass_demo_is_isolated_and_accessible():
    source = DEMO_PATH.read_text(encoding="utf-8")
    parser = _DemoMarkupParser()
    parser.feed(source)

    assert parser.tablist_count == 1
    assert parser.radiogroup_count == 3
    assert len(parser.tabs) == 2
    assert len(parser.radios) == 12
    assert sum(tab.get("aria-selected") == "true" for tab in parser.tabs) == 1
    assert sum(radio.get("aria-checked") == "true" for radio in parser.radios) == 3
    assert all(tab.get("aria-controls") for tab in parser.tabs)
    assert len(parser.optical_switches) == 4
    assert len(parser.liquid_ranges) == 1
    assert len(parser.canvases) == (
        len(parser.optical_switches)
        + len(parser.liquid_ranges)
        + len(parser.toggle_canvases)
        + len(parser.color_canvases)
    ) == 9
    assert len(parser.toggle_canvases) == 3
    assert all(canvas.get("aria-hidden") == "true" for canvas in parser.canvases)
    assert len(parser.switches) == 3
    assert sum(
        switch.get("aria-checked") == "true"
        for switch in parser.switches
    ) == 2
    assert all(switch.get("aria-label") for switch in parser.switches)
    assert len(parser.ranges) == 1
    assert parser.ranges[0].get("aria-label")
    assert parser.ranges[0].get("aria-describedby")
    assert len(parser.motion_blocks) == 4
    assert len(parser.color_canvases) == 1
    assert not parser.external_scripts
    assert "WebSocket" not in source
    assert "fetch(" not in source
    assert "optical-landscape.png" not in source
    assert "BACKGROUND_URL" not in source
    assert "backgroundImage" not in source


def test_liquid_glass_demo_uses_local_scene_refraction_not_blur_effects():
    source = DEMO_PATH.read_text(encoding="utf-8")

    assert 'getContext("webgl2"' in source
    assert "texture(uScene" in source
    assert "sceneAt" in source
    assert "capsuleDistance" in source
    assert "lensAxisHalfPx" in source
    assert "lensAxisPoint" in source
    assert "lensRadialLength" in source
    assert "ellipsePoint" not in source
    assert "edgeBand" in source
    assert "uniform float uPixelRatio;" in source
    assert "uniform float uVelocity;" in source
    assert "uniform float uTrackVisibility;" in source
    assert "trackMask * uTrackVisibility" in source
    assert "trackSignedDistanceAt" in source
    assert "neutralTrackBand" in source
    assert "neutralTrackRefraction" in source
    assert "mix(15.5, 30.0, uInteraction)" in source
    assert "pow(edgeProgress, 1.35)" in source
    assert "blurRadiusPx" in source
    assert "chromaticShiftPx" in source
    assert "redChannel.r" in source
    assert "greenChannel.g" in source
    assert "blueChannel.b" in source
    assert "dispersedLuma" in source
    assert "topFoldBand" in source
    assert "topLineFocus" in source
    assert "topFoldRGB" in source
    assert "chromaticBand" in source
    assert "float glassActivation" in source
    assert "* glassActivation;" in source
    assert "shoulderBand" in source
    assert "directionalLight" in source
    assert "fresnel = pow(edgeProgress, 6.0)" in source
    assert "coreBand" in source
    assert "vec3(0.50)" not in source
    assert "vec3(0.62)" not in source
    assert "adaptiveGray" not in source
    assert "rimBand" not in source
    assert "reflectedPx" in source
    assert "boundaryMappedPx" in source
    assert "interiorContentBand" in source
    assert "contentSampleScale" in source
    assert "compressedContentPx" in source
    assert "topShadowBand" in source
    assert "bottomLightBand" in source
    assert "lowerLightTone" in source
    assert "lowerShadowMask" in source
    assert "lowerEdgeShadowBand" in source
    assert "Three-zone lens" in source
    assert "中央內容輕微縮小" in source
    assert "uTrackHalfSizePx / uLensHalfSizePx" in source
    assert "glassOutputMask" in source
    assert "glassPixels.rgb * glassOutputMask" in source
    assert "texSubImage2D" in source
    assert ": bar.height / 2 + 15;" in source
    assert 'bar.root.dataset.centerOffsetPx = "0"' in source
    assert "drawBackgroundCover" in source
    assert "drawMotionBlocks" in source
    assert "drawColorBlocks" in source
    assert "updateColorBlockLayout" in source
    assert "colorBackgroundCanvas" in source
    assert "context.drawImage(" in source
    assert 'min="8"' in source
    assert 'value="8"' in source
    assert "const CANVAS_PAD_Y = 32;" in source
    assert "const CANVAS_PAD_X = 36;" in source
    assert "quietSelection" in source
    assert 'data.chromaticLayer' not in source
    assert "dataset.chromaticLayer" in source
    assert '? "inner"' in source
    assert "--edge-gray:" in source
    assert "backdrop-filter:" not in source
    assert "filter: blur(" not in source
    assert "infinite" not in source
    assert "rgba(255,255,255" not in source
    assert "convexCore" not in source
    assert "magnifyX" not in source


def test_liquid_glass_demo_has_motion_contrast_and_browser_fallbacks():
    source = DEMO_PATH.read_text(encoding="utf-8")

    assert "prefers-reduced-motion: reduce" in source
    assert "forced-colors: active" in source
    assert "webglcontextlost" in source
    assert "webglcontextrestored" in source
    assert "ResizeObserver" in source
    assert ":focus-visible" in source
    assert 'type="range"' in source
    assert 'data-renderer="css"' in source
    assert "devicePixelRatio" in source
    assert "Math.min" in source
    assert "requestAnimationFrame" in source
    assert "cancelAnimationFrame" in source
    assert "__liquidGlassDiagnostics" in source
    assert 'data-frame-count="0"' in source
    assert 'data-idle="true"' in source
    assert 'data-contexts="0"' in source
    assert "visibilitychange" in source
    assert "Pause motion" in source
    assert "globalState.raf" in source
    assert "maxLensOverflow" in source
    assert "pointerActive" in source
    assert "if (!bar.dragging && pointerDistance <= 6) return;" in source
    assert "bar.root.setPointerCapture?.(event.pointerId);" in source
    assert "const wasDragging = bar.dragging;" in source
    assert "setBarSelection(bar, pressedIndex);" in source
    assert 'event?.type === "pointercancel"' in source
    assert "function startBarTransition" in source
    assert "bar.transitionDuration" in source
    assert "const eased = 1 - Math.pow(1 - progress, 3);" in source
    assert "startBarTransition(bar, 150)" in source
    assert "transitionPulse" in source
    assert "velocityUnit" in source
    assert "function createLiquidToggle" in source
    assert "function createToggleGlassBar" in source
    assert "externalMotion: true" in source
    assert "lazyRenderer: true" in source
    assert "function ensureToggleGlassRenderer" in source
    assert "pendingLazy" in source
    assert "if (bar.externalMotion) return;" in source
    assert 'bar.type === "toggle"' in source
    assert "bar.interaction < 0.002" in source
    assert "dx / 56" in source
    assert "toggle.progress >= 0.5" in source
    assert "!toggle.pointerActive" in source
    assert "delayMs: 120" in source
    assert '"--toggle-glass"' in source
    assert "--toggle-solid-color:" in source
    assert ".liquid-toggle-thumb {" in source
    assert "inset 0 0 0 2px" not in source
    assert "function createLiquidRange" in source
    assert "function updateRangePresentation" in source
    assert "rangeRatio" in source
    assert "lensCenterX" in source
    assert "targetCenterX" in source
    assert "maxCanvasClip" in source
    assert "IntersectionObserver" in source
    assert 'bar.type === "range"' in source
    assert "--canvas-pad-x: 66px;" in source
    assert "? 66" in source
    assert ".liquid-toggle-canvas" in source
