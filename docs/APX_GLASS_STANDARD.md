# APX Liquid Glass Standard

Production UI uses CSS glass primitives, calibrated against
`frontend/static/demos/liquid-glass-switch.html`.

## Component API

```html
<div class="apx-glass-container" data-apx-glass="container"></div>
<div class="apx-glass-nav" data-apx-glass="nav" data-apx-size="md"></div>
<label class="factor-chk apx-factor-switch"></label>
<button class="toggle-field apx-glass-toggle"></button>
<input class="apx-glass-range" data-apx-glass="range" type="range">
<button class="btn apx-glass-action apx-glass-action--green"></button>
```

## Size rule

Do not use fixed large-control pixels on small controls. The demo exposes
effective pixel values with `data-effective-edge-active-px` and
`data-effective-rgb-band-width-px`; production controls should use size tokens:

- `md`: full sidebar/action controls.
- `sm`: compact nav or bottom controls.

## Replacement rule

- Nav bars: use `.apx-glass-nav`; remove underline-only active bars.
- Containers/cards: use `.apx-glass-container`.
- Factor/model binary choices: keep the original checkbox input, wrap with
  `.apx-factor-switch`.
- Sliders: keep native `<input type="range">`, add `.apx-glass-range`.
- Long execution buttons: keep existing IDs/click handlers, add
  `.apx-glass-action` plus an accent modifier.

The rule is: preserve IDs, values, and event handlers; only replace visual
surface classes unless a component requires a dedicated adapter.
