# APX Liquid Glass Standard

Two layers, in this order:

1. **WebGL renderer — `frontend/static/apx-liquid-glass.js`.** The real
   refraction. Same GLSL as the bench in
   `frontend/static/demos/liquid-glass-switch.html`.
2. **CSS glass — `frontend/static/ancserTPX.css`.** The fallback surface when
   WebGL2 is unavailable.

A control that the renderer mounts gets `data-apx-renderer="webgl"`, and the
CSS rules keyed off that attribute switch the CSS fill off so the canvas is the
only visible surface. If WebGL2 is unavailable or the context is lost, nothing
carries that attribute and the CSS glass stays exactly as it is.
`document.documentElement.dataset.apxGlassRenderer` reports which layer is live.

## Component API

```html
<div class="apx-glass-container" data-apx-glass="container"></div>   <!-- CSS -->
<div class="apx-glass-nav" data-apx-glass="nav" data-apx-size="md"></div>
<label class="factor-chk apx-factor-switch"></label>
<button class="toggle-field apx-glass-toggle"></button>
<input class="apx-glass-range" data-apx-glass="range" type="range">
<button class="btn apx-glass-action apx-glass-action--green"></button>
```

Preserve IDs, values, and event handlers; only replace visual surface classes.
`getActiveFactors()`, `getFactorWeights()` and the backend request body are
unchanged by anything in this document.

## What the renderer mounts

| Selector | Type | Lens |
| --- | --- | --- |
| `.apx-glass-nav` | `nav` | selection lens only — no bar, no frame, labels stay in the DOM |
| `.apx-glass-switch input[type=checkbox]` | `toggle` | thumb only — the pill track stays CSS |
| `input[type=range].apx-glass-range` | `range` | thumb over a slim track |
| `.dual-range` | `dual-range` | two thumbs on one shared track (year range) |
| `.panel.apx-glass-container` | `surface` (rounded box) | the whole card — a real glass container, static |
| `.btn.apx-glass-action` | `surface` (rounded box) | whole button |
| `.toggle-field.apx-glass-toggle`, `.conn-trigger.apx-glass-status`, `.account-badge.apx-glass-badge` | `surface` (capsule) | whole control |

The sidebar itself is transparent: the page gradient **is** its background, and
the panels sit on it as glass cards. Panels are static (`hoverActivates: false`)
and only repaint when the layout or the scroll position settles, so a full
36-node repaint costs ~9ms and an idle frame ~0.2ms.

## Interaction rules

- **Idle is flat.** `nav`, `toggle`, `range` and `dual-range` rest at
  interaction `0`: no glass at all, just a solid grey thumb painted into the
  scene, exactly like the bench. The lens fades in as the control is engaged
  and the solid thumb fades out underneath it (`quietWeight`, the port of
  `quietSelection` in `drawBarScene`). Only `surface` controls hold a resting
  glass rim, because they are the page's "glass containers".
- A `surface` never changes shape. Hover raises the lens interaction (rim,
  chroma) and CSS brightens the label; lens growth and content shrink are both
  pinned to zero so the box cannot appear to move or resize.
- A `toggle` uses the demo's spring from `updateLiquidToggle()`: engage
  overshoots to 1.16 and the thumb stretches along travel while squashing
  across it, then decays back. It is the one control that overshoots.
- **Hover never holds a control open, and neither does a click's focus.**
  `nav`, `toggle` and the sliders key off press / drag / travel only
  (`hoverActivates: false`), and focus counts only when it arrived from the
  keyboard (`state.keyboardNav`). Without that second rule a released slider
  stays lit until you click something else, because the input keeps DOM focus.
- Only `surface` controls respond to hover.
- The `nav` and `toggle` thumbs both follow a drag left/right and commit on
  release (the toggle swallows the click the browser fires afterwards so the
  value cannot flip twice). The nav lens swells in **both** axes while engaged,
  using `renderBar()`'s idle/drag half-sizes rather than a flat multiplier.
- Accent colour on a `surface` fills the whole control — no radial falloff that
  leaves one end uncoloured.

## Architecture

The demo opens one WebGL2 context per control. A production page has ~25 glass
controls, past the browser per-page context cap, so the module instead keeps:

- **one** shared WebGL2 context on an offscreen canvas — every control renders
  through it and is blitted into its own 2D canvas;
- one shared viewport-sized **backdrop** canvas (page gradients, then the
  computed `background-color` of `.header` / `.sidebar` / `.bottom-panel` /
  panels) that each control crops as the scene behind its lens;
- one rAF loop that runs only while something is animating and then parks. A
  settled control keeps its last frame at zero cost — one control moving costs
  one render per frame, not 25.

`BACKDROP` and `SCENE` in the module mirror the `body` / `.main::before`
backgrounds and the control fills in `ancserTPX.css`, per theme. **Changing the
page background means changing both**, otherwise the glass refracts a backdrop
the page no longer has.

## Light mode

`setTheme()` in `ancserAPX.js` flips `document.documentElement.dataset.theme`
and persists it; the palette is entirely CSS custom properties under
`:root[data-theme="light"]`. It also re-applies `chartThemeOptions()` to the
Lightweight Chart and calls `ApxGlass.invalidate(true)`, which re-reads the
theme and repaints every backdrop and scene fill. The switch in the header is
itself an `.apx-glass-switch`, so it is rendered by the same toggle adapter as
the factor rows.

Anything styled with a hardcoded colour inline in `ancserAPX.html` (the chart
legend chip, a few inline `background:` attributes) still reads dark in light
mode — those need converting to tokens before light mode is fully clean.

## Lens gotchas

Two traps that both produce "phantom shapes" on screen:

1. The bench's capsule SDF always takes its radius from the **short** axis, so
   a lens taller than it is wide silently collapses into a small circle. Keep
   capsule lenses wider than they are tall (the bench thumb is 65x42); anything
   genuinely upright needs `shape: 'box'` with a `cornerRadius` well under the
   half width, or the rounded box turns back into a capsule.
2. The lens mask is antialiased. A solid idle thumb drawn at exactly the lens
   size leaves a feathered ring of bare backdrop around it, which reads as a
   stray ellipse behind the control. `LENS_AA_COVER_PX` pads the painted thumb
   to cover it.
3. A thumb that swells past its own track refracts whatever the scene holds out
   there — the page backdrop, i.e. a black slab in dark mode and a white one in
   light. The scene must bleed track colour beyond the track
   (`TRACK_BLEED_PX`); the track mask hides the bleed until the lens grows onto
   it.

Lens **size** is eased, not just position (`easeLensSizes`). Neighbouring nav
tabs are different widths, so without it the selection snaps to the new width
the instant the class changes and only the slide is animated.

## Control colours live in `SCENE`, not the CSS

Once a control's canvas owns its surface, its track and thumb colours come from
`SCENE[theme]` in the module — editing the CSS fallback rule changes nothing
that is on screen. Measured thumb-vs-track contrast, which is what makes the
switch readable:

| | OFF | ON |
| --- | --- | --- |
| dark | 7.9 | 2.7 |
| light | 3.5 | 4.6 |

## Follow the bench's layering, don't invent one

The switch is the bench's `.liquid-toggle-control` structure, and that
structure is *why* it never breaks:

```
track  (DOM span, z0)  ── always visible, owns the OFF/ON colour
canvas (z1)            ── lens only, opacity: var(--apx-switch-glass)
thumb  (DOM span, z2)  ── solid, opacity: calc(1 - var(--apx-switch-glass))
input  (z3)            ── transparent, hit target only
```

`syncToggleDom` is the port of `renderLiquidToggle()`: it drives the DOM track
and thumb from the same state the lens uses and cross-fades them, so idle is
pure DOM (canvas at `opacity: 0`) and engaged is pure glass.

**The scene must hand the thumb over as the DOM one fades.** The cross-fade
hides the DOM thumb completely while held, and a lens over a flat one-colour
track refracts nothing visible — so the ball just disappeared until the scene
started painting it at `alpha = glass`. Idle: DOM thumb. Held: scene thumb seen
through the lens. The two are exact complements.

The version where the canvas painted the whole pill broke three times in a row
— covered by the input, then by a theme rule, then leaking backdrop where the
thumb swelled past the track. None of those are possible here, because the
visible surface is never something a stylesheet can paint over.

## Who owns the visible surface

**Verify this by auditing computed style, never by reading canvas pixels.** The
canvas being correct proves nothing — three separate bugs here were "canvas is
right, something covers it". The check that actually catches them:

```js
document.querySelectorAll('[data-apx-renderer="webgl"]').forEach(el => {
    const cs = getComputedStyle(el);          // must be transparent wherever
    console.log(el.className, cs.backgroundColor, cs.boxShadow);  // the canvas
});                                            // owns the surface
```

Run it in **both themes and both control states** — the light-mode OFF switch
was the only combination that broke, because `:root[data-theme="light"] …`
outranks the plain hand-off rule. That is why the hand-off rules use
`!important`.


Where a control is both the hit target and the thing being styled (the switch
input), the input sits **above** the canvas so it can still be clicked — which
means its own CSS background would hide everything the renderer paints. Those
controls run with `trackVisibility: 1` so the canvas paints the whole pill, and
the CSS fill is switched off under `[data-apx-renderer="webgl"]`. Getting this
backwards is invisible to canvas pixel checks: the canvas is correct, it is
just covered.

## Shader fidelity

The demo is the calibration source. Do not hand-edit the GLSL inside
`apx-liquid-glass.js` — edit the demo, then run:

```
python scripts/sync_glass_shader.py
```

The only allowed deviation is the `uCornerRadiusPx` uniform, which lets a
production surface pick its corner radius instead of the demo's hardcoded
`0.31 * half-size`; at `0` it reproduces the demo exactly.
`tests/test_apx_glass_standard.py` fails if the two drift apart.

## Size rule

Do not use large-control pixel values on small controls. Every node scales the
demo's px tuning by `control height / 58px` (the demo's md switch height), so a
20px factor toggle gets ~0.34× the edge width and RGB band of a full-size
control. Size tokens (`data-apx-size="md" | "sm"`) remain the CSS-side hint.

## QA

`window.ApxGlass.diagnostics` reports `renderer`, `nodes`, `contexts`,
`contextFailures`, `frames`, `renders`, `idle`, and per-node lens state.
`window.ApxGlass.step(frames)` advances the renderer without rAF — needed
because rAF is paused while the tab is hidden, which is the usual state in a
headless check.

## Stacking

The canvas is a positioned element inside the control's mount, so anything
positioned later in the document paints over it. Two cases bit us: the sticky
`<thead>` (`z-index: 3`) covering the bottom nav's lens, and a control that is
its own hit target (the switch input) covering its canvas. Neither shows up in
a canvas pixel check — the canvas is correct, it is just underneath something.
