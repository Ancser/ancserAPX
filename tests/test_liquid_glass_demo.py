from html.parser import HTMLParser
from pathlib import Path


DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "static"
    / "demos"
    / "liquid-glass-switch.html"
)
DEMO_ASSETS_PATH = DEMO_PATH.parent / "assets"


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
        self.liquid_squares = []
        self.motion_blocks = []
        self.external_scripts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if self.hidden_depth:
            self.hidden_depth += 1
            return
        if "hidden" in attributes:
            self.hidden_depth = 1
            return
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
        if "data-liquid-square" in attributes:
            self.liquid_squares.append(attributes)
        if "data-motion-block" in attributes:
            self.motion_blocks.append(attributes)
        if tag == "script" and attributes.get("src"):
            self.external_scripts.append(attributes["src"])

    def handle_endtag(self, tag):
        if self.hidden_depth:
            self.hidden_depth -= 1


def test_liquid_glass_demo_is_isolated_and_accessible():
    source = DEMO_PATH.read_text(encoding="utf-8")
    parser = _DemoMarkupParser()
    parser.feed(source)

    assert parser.tablist_count == 6
    assert parser.radiogroup_count == 0
    assert len(parser.tabs) == 12
    assert len(parser.radios) == 0
    assert sum(tab.get("aria-selected") == "true" for tab in parser.tabs) == 6
    assert all(tab.get("aria-controls") for tab in parser.tabs)
    assert len(parser.optical_switches) == 6
    assert len(parser.liquid_ranges) == 6
    assert len(parser.liquid_squares) == 1
    assert len(parser.canvases) == (
        len(parser.optical_switches)
        + len(parser.liquid_ranges)
        + len(parser.liquid_squares)
        + len(parser.toggle_canvases)
        + len(parser.color_canvases)
    ) == 20
    assert len(parser.toggle_canvases) == 6
    assert all(canvas.get("aria-hidden") == "true" for canvas in parser.canvases)
    assert len(parser.switches) == 6
    assert sum(
        switch.get("aria-checked") == "true"
        for switch in parser.switches
    ) == 6
    assert all(switch.get("aria-label") for switch in parser.switches)
    assert len(parser.ranges) == 6
    assert all(range_input.get("aria-label") for range_input in parser.ranges)
    assert parser.ranges[0].get("aria-describedby")
    assert sum(
        item.get("data-demo-active") == "true"
        for item in [*parser.optical_switches, *parser.liquid_ranges, *parser.switches]
    ) == 10
    assert len(parser.motion_blocks) == 0
    assert len(parser.color_canvases) == 1
    assert not parser.external_scripts
    assert "WebSocket" not in source
    assert "fetch(" not in source
    assert "optical-landscape.png" not in source
    assert "BACKGROUND_URL" not in source
    assert "backgroundImage" not in source


def test_liquid_glass_demo_uses_local_scene_refraction_not_blur_effects():
    source = DEMO_PATH.read_text(encoding="utf-8")

    assert (DEMO_ASSETS_PATH / "kube-switch-displacement.png").is_file()
    assert (DEMO_ASSETS_PATH / "kube-switch-specular.png").is_file()
    assert 'getContext("webgl2"' in source
    assert "texture(uScene" in source
    assert "sceneAt" in source
    assert "capsuleDistance" in source
    assert "roundedBoxDistance" in source
    assert "lensDistanceAt" in source
    assert "lensDistanceNormal" in source
    assert "uniform float uShapeMode;" in source
    assert "lensAxisHalfPx" in source
    assert "lensAxisPoint" in source
    assert "lensRadialLength" in source
    assert "ellipsePoint" not in source
    assert "edgeBand" in source
    assert "uniform float uPixelRatio;" in source
    assert "uniform float uVelocity;" in source
    assert "uniform float uTrackVisibility;" in source
    assert "uniform vec4 uShapeTuning;" in source
    assert "uniform vec4 uInnerTuning;" in source
    assert "uniform vec4 uBlendTuning;" in source
    assert "uniform vec4 uEdgeTuning;" in source
    assert "uniform vec4 uRimTuning;" in source
    assert "uniform vec4 uRoundTuning;" in source
    assert "uniform vec4 uTransitionTuning;" in source
    assert "uniform vec4 uOpticTuning;" in source
    assert "uniform vec4 uLightTuning;" in source
    assert "uniform vec4 uKubeTuning;" in source
    assert "trackMask * uTrackVisibility" in source
    assert "trackSignedDistanceAt" in source
    assert "neutralTrackBand" in source
    assert "neutralTrackRefraction" in source
    assert "mix(uShapeTuning.x, uShapeTuning.y, uInteraction)" in source
    assert "pow(edgeProgress, 1.35)" in source
    assert "blurRadiusPx" in source
    assert "uEdgeTuning.x * uPixelRatio" in source
    assert "chromaticShiftPx" in source
    assert "redChannel.r" in source
    assert "greenChannel.g" in source
    assert "blueChannel.b" in source
    assert "rimFoldBand" in source
    assert "rimFoldBand * uBlendTuning.w * uInteraction" in source
    assert "rimFoldProgress" in source
    assert "centerFoldPx" in source
    assert "foldedContent" in source
    assert "chromaticBand" in source
    assert "float glassActivation" in source
    assert "* glassActivation;" in source
    assert "shoulderBand" in source
    assert "directionalLight" in source
    assert "fresnel = pow(edgeProgress, 6.0)" in source
    assert "fresnelRim" in source
    assert "ambientGlassFill" in source
    assert "darkCompensation" in source
    assert "specularLobe" in source
    assert "innerRimBand" in source
    assert "oppositeEdgeOcclusion" in source
    assert "coreBand" in source
    assert "vec3(0.50)" not in source
    assert "vec3(0.62)" not in source
    assert "adaptiveGray" not in source
    assert "rimBand" not in source
    assert "reflectedPx" in source
    assert "boundaryMappedPx" in source
    assert "interiorContentBand" in source
    assert "contentSampleScale" in source
    assert "magnificationScale" in source
    assert "uShapeTuning.z" in source
    assert "floatingLensWeight * uShapeTuning.w" in source
    assert "compressedContentPx" in source
    assert "floatingLensWeight" in source
    assert "innerEdgeProgress" in source
    assert "transitionWidthPx" in source
    assert "transitionProgress" in source
    assert "transitionDecay" in source
    assert "transmissionMix" in source
    assert "opticalTransmission" in source
    assert "ambientFillStrength" in source
    assert "exp(-transitionLinear * transitionDecay)" in source
    assert "rgbBandWidthPx" in source
    assert "rgbHaloBand" in source
    assert "rgbHaloProgress" in source
    assert "chromaticBasePx" in source
    assert "chromaticBasePx - compressedContentPx" in source
    assert "sceneAt(chromaticBasePx)" in source
    assert "edgeTaperPower" in source
    assert "edgeOpticBand" in source
    assert "edgeRefractionRamp" in source
    assert "edgeRefractionStrength" in source
    assert "kubeRefractionLevelPx" in source
    assert "max(uKubeTuning.x, 0.0) * 22.0 * uPixelRatio" in source
    assert "edgeFoldCurve" in source
    assert "sideRoundPull" in source
    assert "innerRefractionPullPx" in source
    assert "uInnerTuning.z * mix(0.72, 1.0, uStrength)" in source
    assert "floatingLensWeight * uInnerTuning.w" in source
    assert "innerWarpedContentPx" in source
    assert "innerWarpedContent" in source
    assert "edgeCrispness" in source
    assert "clearRimBand" in source
    assert "-uRimTuning.x * uPixelRatio" in source
    assert "clearRimTone" in source
    assert "rimRoundness" in source
    assert "topShadowBand" in source
    assert "bottomLightBand" in source
    assert "lowerLightTone" in source
    assert "topSidePull" not in source
    assert "dispersedLuma" not in source
    assert "topFoldBand" not in source
    assert "topLineFocus" not in source
    assert "topFoldRGB" not in source
    assert "lowerShadowMask" not in source
    assert "lowerEdgeShadowBand" not in source
    assert "Three-zone lens" not in source
    assert "Shared background optics" not in source
    assert "Edge only." not in source
    assert "Scene aware." not in source
    assert "Solid until press" not in source
    assert "Neutral outer glass" not in source
    assert "Risk posture" not in source
    assert 'label: "ALPHA"' not in source
    assert "uTrackHalfSizePx / uLensHalfSizePx" in source
    assert "data-liquid-square" in source
    assert "createSquareGlass" in source
    assert 'bar.type === "square"' in source
    assert "drawSquareSceneText" in source
    assert "PLACEHOLDER TEXT" in source
    assert "REFRACTION" in source
    assert "RESULT" in source
    assert "EDGE SAMPLE" in source
    assert "shapeMode: gl.getUniformLocation" in source
    assert "glassOutputMask" not in source
    assert "glassPixels.rgb * outputMask" in source
    assert "glassLabelIsolation" not in source
    assert '[data-glass-active="true"] .optical-switch-button:not' not in source
    assert "texSubImage2D" in source
    assert ": bar.height / 2 + 15;" in source
    assert 'bar.root.dataset.centerOffsetPx = "0"' in source
    assert "drawBackgroundCover" in source
    assert "drawMotionBlocks" not in source
    assert "data-motion-block" not in source
    assert "optical-motion-field" not in source
    assert "optical-motion-block" not in source
    assert "drawColorBlocks" in source
    assert "updateColorBlockLayout" in source
    assert "colorBackgroundCanvas" in source
    assert "context.drawImage(" in source
    assert 'min="8"' in source
    assert 'value="52"' in source
    assert 'value="58"' in source
    assert "const CANVAS_PAD_Y = 32;" in source
    assert "const CANVAS_PAD_X = 36;" in source
    assert "quietSelection" in source
    assert 'id="optical-tuning-grid"' in source
    assert 'id="optical-tuning-reset"' in source
    assert "const tuningControls = [" in source
    assert "renderTuningControls()" in source
    assert "resetTuningControls" in source
    assert "Center-to-edge transition is mainly Transition spread/decay" not in source
    assert "RGB band width is separate from the final 3px edge rim." not in source
    assert "optical-tuning-help" not in source
    assert "optical-bench-preview" in source
    assert "optical-bench-set--held" in source
    assert "optical-bench-side--light" in source
    assert "optical-bench-side--dark" in source
    assert 'data-demo-active="true"' in source
    assert 'data-apx-size="sm"' in source
    assert "APX_SIZE_PRESETS" in source
    assert "resolveApxSize" in source
    assert "sizeScale" in source
    assert "effectiveEdgeActivePx" in source
    assert "effectiveRgbBandWidthPx" in source
    assert "tuning.edgeActive * sizeScale" in source
    assert "tuning.rgbBandWidth * sizeScale" in source
    assert "demoHoldActive" in source
    assert "sceneSide" in source
    assert 'bar?.sceneSide === "light"' in source
    assert 'bar?.sceneSide === "dark"' in source
    assert 'id="optical-strength-output"' in source
    assert "optical-tuning-control--pair" in source
    assert "optical-tuning-range-pair" in source
    assert "const tuningLayout = [" in source
    assert 'label: "Inner mix"' in source
    assert 'keys: ["innerMixMin", "innerMixMax"]' in source
    assert 'label: "RGB mix"' in source
    assert 'keys: ["chromaMin", "chromaMax"]' in source
    assert "control.help" not in source
    assert "overflow: visible;" in source
    assert "max-height: min(42vh, 390px)" not in source
    assert "input.dataset.tuningParam = control.key" in source
    assert '"edgeActive"' in source
    assert 'label: "Edge width active"' in source
    assert 'value: 3,' in source
    assert 'value: 24,' in source
    assert '"transitionSpread"' in source
    assert 'label: "Transition spread"' in source
    assert 'value: 42,' in source
    assert "Independent center-to-edge shoulder width" not in source
    assert '"transitionDecay"' in source
    assert 'label: "Transition decay"' in source
    assert '"transmission"' in source
    assert 'label: "Transmission"' in source
    assert '"ambientFill"' in source
    assert 'label: "Dark fill"' in source
    assert '"magnification"' in source
    assert 'label: "Center lens"' in source
    assert '"edgeRefraction"' in source
    assert 'label: "Edge refraction"' in source
    assert '"refractionLevel"' in source
    assert 'label: "Refraction level"' in source
    assert 'value: 1,' in source
    assert '"specularOpacity"' in source
    assert 'label: "Specular opacity"' in source
    assert '"specularSaturation"' in source
    assert 'label: "Specular saturation"' in source
    assert '"blurLevel"' in source
    assert 'label: "Blur level"' in source
    assert '"progressiveBlur"' in source
    assert 'label: "Progressive blur"' in source
    assert '"glassBackgroundOpacity"' in source
    assert 'label: "Glass bg opacity"' in source
    assert "Exponential falloff" not in source
    assert '"rgbBandWidth"' in source
    assert 'label: "RGB band width"' in source
    assert 'value: 36,' in source
    assert "RGB coverage width independent from Edge width active." not in source
    assert '"edgeTaper"' in source
    assert 'label: "Edge taper"' in source
    assert "Nonlinear rim falloff" not in source
    assert '"edgeBlur"' not in source
    assert '"innerPull"' not in source
    assert '"floatingPull"' not in source
    assert 'label: "Warp pull"' not in source
    assert '"centerShrink"' in source
    assert '"rimStrength"' in source
    assert '"fresnelStrength"' in source
    assert 'label: "Fresnel rim"' in source
    assert '"specularStrength"' in source
    assert 'label: "Specular"' in source
    assert '"innerRimStrength"' in source
    assert 'label: "Inner rim"' in source
    assert '"darkRimStrength"' in source
    assert 'label: "Dark rim"' in source
    assert '"motionStretch"' in source
    assert 'label: "Motion stretch"' in source
    assert "How deep the center-to-edge transition starts." not in source
    assert "Content distortion before it reaches the rim." not in source
    assert "Higher value makes center-to-edge blend stronger." not in source
    assert "Delay center color folding into rim." not in source
    assert "shapeTuning: gl.getUniformLocation" in source
    assert "transitionTuning: gl.getUniformLocation" in source
    assert "opticTuning: gl.getUniformLocation" in source
    assert "lightTuning: gl.getUniformLocation" in source
    assert "kubeTuning: gl.getUniformLocation" in source
    assert "kubeMaterialTuning: gl.getUniformLocation" in source
    assert "tuning.refractionLevel * sizeScale" in source
    assert "tuning.progressiveBlur * sizeScale" in source
    assert "tuning.blurLevel * sizeScale" in source
    assert "tuning.glassBackgroundOpacity" in source
    assert "tuning.specularOpacity" in source
    assert "tuning.specularSaturation" in source
    assert "uniform vec4 uKubeMaterialTuning;" in source
    assert "kubeSpecularOpacity" in source
    assert "kubeSpecularSaturation" in source
    assert "kubeBlurLevelPx" in source
    assert "kubeGlassBackgroundOpacity" in source
    assert "kubeProgressiveBlurPx" in source
    assert "magnificationScale = min(" in source
    assert "refractionBridgeProgress" in source
    assert "refractionBridgeBand" in source
    assert "solidRefractionBand" in source
    assert "gl.uniform4f(" in source
    assert "* 0.34" in source
    assert "* 0.46" in source
    assert "mix(0.32, 0.78, opticalTransmission)" in source
    assert "* 1.18" not in source
    assert "#30d158" in source
    assert "rgb(48 209 88)" in source
    assert "colorA:" not in source
    assert "colorB:" not in source
    assert "colorBlockSpecs" not in source
    assert "colorBlockRects" not in source
    assert "rgb(242 242 238)" in source
    assert "rgb(5 5 5)" in source
    assert "previewRect" in source
    assert "verticalSplit" in source
    assert "rgb: [66, 68, 74]" not in source
    assert "rgb: [44, 47, 54]" not in source
    assert "colorA: [255, 77, 138]" not in source
    assert "colorA: [244, 196, 0]" not in source
    assert "colorA: [42, 106, 229]" not in source
    assert 'data.chromaticLayer' not in source
    assert "dataset.chromaticLayer" in source
    assert '? "inner"' in source
    assert "--edge-gray:" in source
    assert "Implementation split" not in source
    assert 'data-implementation="' not in source
    assert "id=\"kube-refraction-filter\"" in source
    assert "id=\"kube-displacement-image\"" in source
    assert "id=\"kube-refraction-map\"" in source
    assert "id=\"kube-thumb-filter\"" in source
    assert "assets/kube-switch-displacement.png" in source
    assert "assets/kube-switch-specular.png" in source
    assert "kube-thumb-displacement" in source
    assert "kube-thumb-saturation" in source
    assert "kube-thumb-specular-alpha" in source
    assert "feDisplacementMap" in source
    assert "feSpecularLighting" in source
    assert "kube-magnifier" in source
    assert "kube-search" in source
    assert "kube-nav" in source
    assert "kube-player" in source
    assert "kube-container" in source
    assert "kube-switch" in source
    assert "kube-slider" in source
    assert "Specular opacity" in source
    assert "Specular saturation" in source
    assert "Refraction level" in source
    assert "Blur level" in source
    assert "Glass bg opacity" in source
    assert "--kube-refraction-level" in source
    assert "--kube-blue-opacity" in source
    assert "--kube-glass-background-opacity" in source
    assert "--kube-specular-opacity" in source
    assert "--kube-progressive-blur" in source
    assert "kubeSmootherstep" in source
    assert "kubeConvexSquircle" in source
    assert "buildKubeDisplacementMap" in source
    assert "updateKubeDisplacementMap" in source
    assert "stageDisplacement?.setAttribute" in source
    assert "state.refractionLevel * 50.08645714048877" in source
    assert "backdrop-filter:" in source
    assert "url(\"#kube-refraction-filter\")" in source
    assert "url(\"#kube-thumb-filter\")" in source
    assert "initKubeSettings" in source
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
    assert "Pause motion" not in source
    assert "Resume motion" not in source
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
    assert "const springBase = 1 - Math.pow(1 - progress, 3);" in source
    assert "const springOvershoot" in source
    assert "springBase + springOvershoot" in source
    assert "startBarTransition(bar, 150)" in source
    assert "transitionPulse" in source
    assert "motionStretch" in source
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
    assert "dx / toggle.travelPx" in source
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
    assert "apxSize.rangePadXPx" in source
    assert ".liquid-toggle-canvas" in source
