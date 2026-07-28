from html.parser import HTMLParser
from pathlib import Path


GALLERY_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "static"
    / "demos"
    / "liquid-glass-component-gallery.html"
)


class _GalleryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external_scripts = []
        self.ranges = []
        self.exhibits = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.external_scripts.append(attributes["src"])
        if tag == "input" and attributes.get("type") == "range":
            self.ranges.append(attributes)
        if attributes.get("data-component-card"):
            self.exhibits.append(attributes["data-component-card"])


def test_component_lab_is_standalone_and_uses_physical_refraction_core():
    source = GALLERY_PATH.read_text(encoding="utf-8")
    parser = _GalleryParser()
    parser.feed(source)

    assert not parser.external_scripts
    assert "fetch(" not in source
    assert "WebSocket" not in source
    for token in [
        "physicalProfile",
        "createShrinkMap",
        "createDisplacementMap",
        "createSpecularMap",
        "createOpticalFilter",
        "feDisplacementMap",
        "xChannelSelector=\"R\"",
        "yChannelSelector=\"G\"",
        "buildOpticalSurfaces",
        "syncOpticalSurfaces",
        "clip-path: inset(0 round 999px)",
    ]:
        assert token in source


def test_component_lab_contains_only_requested_experiments():
    source = GALLERY_PATH.read_text(encoding="utf-8")
    parser = _GalleryParser()
    parser.feed(source)

    expected = {
        "precision",
        "slider",
        "switch",
        "dock",
        "segment",
        "input",
        "volume",
        "fab",
        "stepper",
    }
    assert set(parser.exhibits) == expected

    for name in [
        "Precision Lens",
        "Fluid Slider",
        "Tactile Switch",
        "Drag Dock",
        "Segment Control",
        "Fluid Input",
        "Squish Volume",
        "Layered FAB",
        "Magnetic Stepper",
    ]:
        assert name in source

    for removed in [
        "Music Player",
        "Data Table / Stats",
        "Weather Widget",
        "Calendar / Date Picker",
        "Kanban / Surface Layout",
        "All 70 source components",
        "componentGroups",
    ]:
        assert removed not in source


def test_each_experiment_has_independent_optics_and_motion_tuning():
    source = GALLERY_PATH.read_text(encoding="utf-8")

    for component in [
        "precision",
        "slider",
        "switch",
        "dock",
        "segment",
        "input",
        "volume",
        "fab",
        "stepper",
    ]:
        assert f"{component}: {{ profile:" in source

    for setting in [
        "profile",
        "bezel",
        "refraction",
        "thickness",
        "shrink",
        "specular",
        "blur",
        "saturation",
        "idleScale",
        "activeScale",
        "stiffness",
        "damping",
        "stretch",
    ]:
        assert f'key: "{setting}"' in source

    for interaction in [
        "initPrecisionLens",
        "initFluidSlider",
        "initTactileSwitch",
        "initDragDock",
        "initSegmentControl",
        "initFluidInput",
        "initSquishVolume",
        "initLayeredFab",
        "initMagneticStepper",
        "runSpringLoop",
        "activeSpringLoops",
        "fastReturn",
    ]:
        assert interaction in source

    assert ".glass::after" not in source
    assert "linear-gradient(120deg, transparent 35%" not in source


def test_center_shrink_and_active_scale_presets_match_component_roles():
    source = GALLERY_PATH.read_text(encoding="utf-8")

    for token in [
        'key: "shrink", label: "Center shrink"',
        'key: "activeScale", label: "Active scale", min: 0.6, max: 2',
        'precision: { profile: "convex-squircle", bezel: 30, refraction: 1.50, thickness: 150, shrink: 0.00',
        'slider: { profile: "convex-squircle", bezel: 16, refraction: 0.85, thickness: 80, shrink: 0.00',
        'switch: { profile: "convex-squircle", bezel: 19, refraction: 0.90, thickness: 47, shrink: 0.30',
        'dock: { profile: "convex-squircle", bezel: 18, refraction: 0.92, thickness: 70, shrink: 0.30',
        'segment: { profile: "convex-squircle", bezel: 18, refraction: 0.88, thickness: 68, shrink: 0.30',
        'volume: { profile: "convex-squircle", bezel: 22, refraction: 0.82, thickness: 90, shrink: 0.00',
        "activeScale: 1.50",
        'result="shrinkMap"',
        'in="shrunk" in2="displacementMap"',
        'filter.setAttribute("x", "-100%")',
        'filter.setAttribute("width", "300%")',
        "const dx = (x - width / 2) * zoomOut;",
        "const dy = (y - height / 2) * zoomOut;",
        "optical-background-copy",
        "optical-content-copy",
        "contentFilterNodes",
        "configureNodes(surface.contentFilterNodes, 0, 0)",
        'root.classList.toggle("interacting", dragging)',
        'track.classList.toggle("interacting", glassShapeActive)',
        "returningFlat",
        'track.style.setProperty("--switch-thumb-color", "rgb(255 255 255)")',
    ]:
        assert token in source

    assert source.count("activeScale: 1.50") == 3


def test_interactive_components_share_live_geometry_without_sampling_clones():
    source = GALLERY_PATH.read_text(encoding="utf-8")

    for token in [
        "logicalOffsetWithin",
        "syncDynamicSample",
        '.optical-stage-copy [data-optical]',
        "const buttons = Array.from(dock.children)",
        "const buttons = Array.from(track.children)",
        "const actions = Array.from(main.parentElement.children)",
        'bubble.style.left = `${7 + x.value}px`',
        'indicator.style.left = `${4 + x.value}px`',
        'action.style.bottom = `${14 + distance * p}px`',
    ]:
        assert token in source

    assert 'dock.querySelectorAll("[data-dock-index]")' not in source
    assert 'track.querySelectorAll("[data-segment-index]")' not in source
    assert 'document.querySelectorAll("[data-fab-action]")' not in source


def test_slider_and_switch_use_continuous_direct_interaction_state():
    source = GALLERY_PATH.read_text(encoding="utf-8")

    for token in [
        "thumbX <= edgeEpsilon",
        "? root.clientWidth",
        '"endpoint-transition"',
        'fill.style.width = `${fillWidth}px`',
        'thumb.style.left = `${thumbX}px`',
        'syncOpticalSurfaces("slider")',
        'surface.stageCopy.querySelector(".slider-fill")',
        'surface.stageCopy.querySelector(".switch-track")',
        'surface.stageCopy.querySelector(".volume-fill")',
        "let pointerActive = false",
        "let dragging = false",
        "startProgress + dx / Math.max(1, travel())",
        'track.classList.toggle("interacting", glassShapeActive)',
        "--switch-track-color",
        "--switch-thumb-color",
        "--switch-glass",
        "--slider-glass",
        ".switch-track.interacting .switch-thumb .optical-layer",
        '"lostpointercapture"',
        '"dragstart"',
        "pointerId !== event.pointerId",
        "pointerActive || positionMoving",
    ]:
        assert token in source


def test_dock_drag_threshold_and_full_capsule_volume():
    source = GALLERY_PATH.read_text(encoding="utf-8")

    for token in [
        "const dragDelayMs = 140",
        "const dragDistancePx = 10",
        "elapsed < dragDelayMs",
        "ignoreClickUntil",
        'class="volume-reservoir"',
        'class="volume-glass-fill"',
        'role="slider"',
        'fill.style.height = `${value * 100}%`',
        'glassFill.style.height = `${value * 100}%`',
        'zone.setAttribute("aria-valuenow"',
        'event.key !== "ArrowUp"',
        'reservoir.style.transform = transform',
        'sampleReservoir.style.transform = sourceReservoir.style.transform',
        'dock.style.setProperty("--control-glass"',
        'track.style.setProperty("--control-glass"',
        ".dock-track.interacting .dock-bubble",
        ".segment-track.interacting .segment-indicator",
        'class="control-source-content"',
        "cutOriginalContentUnderLens",
        'const clip = `path(evenodd, "${outer} ${hole}")`',
        "content.style.clipPath = clip",
        "content.style.webkitClipPath = clip",
    ]:
        assert token in source

    assert ".optical-stage-copy .dock-track button" not in source
    assert ".optical-stage-copy .segment-track button" not in source
    assert "button.style.opacity = (1 - coverage).toFixed(4)" not in source

    assert 'class="volume-line"' not in source
