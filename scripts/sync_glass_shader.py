"""Sync the production liquid-glass GLSL from the demo bench.

`frontend/static/demos/liquid-glass-switch.html` stays the calibration source
for the shader. `frontend/static/apx-liquid-glass.js` must carry the same GLSL
so the production UI refracts exactly like the bench, with one documented
deviation: production controls need explicit corner radii, so an extra
`uCornerRadiusPx` uniform overrides the demo's hardcoded 0.31 * half-size rule
when it is greater than zero.

Run after editing the demo shader:

    python scripts/sync_glass_shader.py

`tests/test_apx_glass_standard.py` calls `build_production_shaders()` to fail
the suite when the two files drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "frontend" / "static" / "demos" / "liquid-glass-switch.html"
MODULE_PATH = ROOT / "frontend" / "static" / "apx-liquid-glass.js"

VERTEX_MARKER = "/*__APX_VERTEX_SHADER__*/"
FRAGMENT_MARKER = "/*__APX_FRAGMENT_SHADER__*/"

_UNIFORM_ANCHOR = "                uniform float uShapeMode;\n"
_UNIFORM_PATCH = _UNIFORM_ANCHOR + "                uniform float uCornerRadiusPx;\n"

_RADIUS_ANCHOR = """                float lensDistanceAt(vec2 point) {
                    float squareRadius = min(
                        min(uLensHalfSizePx.x, uLensHalfSizePx.y) * 0.31,
                        42.0 * uPixelRatio
                    );
"""
_RADIUS_PATCH = """                float lensDistanceAt(vec2 point) {
                    float autoRadius = min(
                        min(uLensHalfSizePx.x, uLensHalfSizePx.y) * 0.31,
                        42.0 * uPixelRatio
                    );
                    float squareRadius = uCornerRadiusPx > 0.0
                        ? min(
                            uCornerRadiusPx,
                            min(uLensHalfSizePx.x, uLensHalfSizePx.y)
                        )
                        : autoRadius;
"""


def _extract(source: str, name: str) -> str:
    marker = f"const {name} = `"
    start = source.index(marker) + len(marker)
    end = source.index("`", start)
    return source[start:end]


def build_production_shaders() -> Tuple[str, str]:
    """Return (vertex, fragment) GLSL the production module must contain."""
    demo = DEMO_PATH.read_text(encoding="utf-8")
    vertex = _extract(demo, "vertexShaderSource")
    fragment = _extract(demo, "fragmentShaderSource")

    for source in (vertex, fragment):
        if "`" in source or "${" in source:
            raise ValueError("shader source cannot be embedded in a JS template literal")

    if _UNIFORM_ANCHOR not in fragment:
        raise ValueError("demo shader no longer declares uShapeMode as expected")
    if _RADIUS_ANCHOR not in fragment:
        raise ValueError("demo shader no longer computes squareRadius as expected")

    fragment = fragment.replace(_UNIFORM_ANCHOR, _UNIFORM_PATCH, 1)
    fragment = fragment.replace(_RADIUS_ANCHOR, _RADIUS_PATCH, 1)
    return vertex, fragment


def _replace_shader_const(module: str, const_name: str, shader: str) -> str:
    pattern = rf"const {const_name} = `[\s\S]*?`;"
    replacement = f"const {const_name} = `{shader}`;"
    updated, count = re.subn(pattern, replacement, module, count=1)
    if count != 1:
        raise ValueError(f"could not replace {const_name}")
    return updated


def main() -> int:
    vertex, fragment = build_production_shaders()
    module = MODULE_PATH.read_text(encoding="utf-8")
    if VERTEX_MARKER in module and FRAGMENT_MARKER in module:
        module = module.replace(VERTEX_MARKER, vertex, 1)
        module = module.replace(FRAGMENT_MARKER, fragment, 1)
    elif (
        "const VERTEX_SHADER_SOURCE = `" in module
        and "const FRAGMENT_SHADER_SOURCE = `" in module
    ):
        module = _replace_shader_const(
            module,
            "VERTEX_SHADER_SOURCE",
            vertex,
        )
        module = _replace_shader_const(
            module,
            "FRAGMENT_SHADER_SOURCE",
            fragment,
        )
    else:
        print(
            "apx-liquid-glass.js has no shader markers or shader constants "
            "that can be synced."
        )
        return 1
    MODULE_PATH.write_text(module, encoding="utf-8")
    print(f"synced vertex ({len(vertex)} chars) and fragment ({len(fragment)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
