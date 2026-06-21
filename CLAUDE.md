# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

OperaMind is a static marketing website for a "digital AI workforce" agency. The current codebase is **two standalone HTML files** with no build system, no framework, and no dependencies — everything (CSS, JS, markup) is inline.

- `index.html` — the main marketing landing page (1 419 lines)
- `agents.html` — the full catalog of 7 AI agents, linked from the nav

Both files are directly openable in a browser; there is no dev server, no bundler, no npm.

## Architecture

### CSS design system (inline in both files)

Both files share the same CSS custom properties defined in `:root`. Any color, font, or spacing change must be made in **both files**. Key tokens:

| Token | Purpose |
|---|---|
| `--bg / --bg1 / --bg2` | Dark backgrounds (layered depth) |
| `--blue / --bluel / --blued` | Primary brand blue |
| `--cyan / --violet / --violetm` | Accent colors |
| `--t1 / --t2 / --t3` | Text hierarchy (light → muted → faint) |
| `--display` | Sora — headings |
| `--mono` | JetBrains Mono — labels, stats, code-like UI |
| `--body` | Inter — body copy |

Gradient text is applied via `.gtext` (blue→violet) and `.mtext` (blue→cyan). Never use `color` on gradient-text elements — only `-webkit-text-fill-color`.

### Layout classes

- `.W` — max-width 1200px centered container
- `.WL` — max-width 1340px (wide variant, index.html only)
- `section` — always 120px vertical padding (80px on mobile)

### Component patterns

- **Pills** (`.pill.pill-blue/.pill-green/.pill-violet`) — mono uppercase labels with animated `.pdot`
- **Buttons** (`.btn`) — combined with size (`.btn-lg`, `.btn-xl`) and style (`.btn-p` primary gradient, `.btn-g` ghost, `.btn-green`)
- **Cards** — `.wcard` (workforce), `.pkg` (pricing), `.tc` (testimonials), `.ind-card` (industries) all follow the same border/card-background/hover-lift pattern
- **Stat displays** — `.dstat-n`, `.kpi-n`, `.ps .n` all use the display font at large weight; gradient variant uses `.g` class

### JavaScript (vanilla, inline at bottom of index.html)

Three self-contained functions:

1. **Nav scroll** — toggles `.scrolled` class on `#mainNav` past 20px scroll
2. **ROI Calculator** (`calcROI()`) — reads four `<input type=range>` values, computes annual savings using a fixed cost table `[0,297,597,997,1290,1590,1890,2190]` for 0–7 agents, updates DOM. Called on every slider `input` event and once on load.
3. **AI Scanner** (`runScan()`) — maps `(industry, size, pain)` → hardcoded opportunity strings; simulates a scan with `setTimeout` + CSS class toggle (`.scanner-results.show`). The "results" are entirely client-side — no API call.

### Animations

All animations are CSS keyframes: `blink`, `float`, `marquee`, `fadeUp`, `countUp`. The `.fu` class + `.d1`–`.d4` delay classes handle entrance animations. `@media(prefers-reduced-motion:reduce)` disables all animations globally.

### Responsive breakpoints

- `≤1024px` — single-column hero, problem grid, calc
- `≤860px` — hamburger nav, single-column cards/packages/testimonials
- `≤580px` — stacked hero CTAs, 1-col KPIs and footer

## Key constraints

- The Calendly booking link in the CTA section (`#cta`) is a placeholder (`https://calendly.com`) and needs a real URL.
- The contact email `hello@operamind.ai` appears in the CTA section.
- `agents.html` is linked from nav, footer, and workforce card CTAs — it must remain at the same relative path.
- The ticker (`.ttrack`) uses `marquee` animation and is duplicated (`aria-hidden="true"` on the copy) to create the infinite scroll illusion — do not add items to only one copy.
- SVG gradient IDs (`ng1`, `ng2` in nav, `fg1`, `fg2` in footer) must remain unique per file; they are inlined and would conflict if the files were ever concatenated.
