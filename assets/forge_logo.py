"""Generate the FERAL forged-metal wordmark SVG.

Usage: python assets/forge_logo.py OUT.svg [--bg white|dark|none]
                                            [--layout wide|square|icon]

wide is the 1500x520 wordmark; square centres the wordmark on a
1024x1024 canvas; icon is a 1024x1024 forged F monogram.
"""

import math
import random
import sys

DY = 70  # vertical offset of the cap line

# Chiseled serif glyphs, cap height 300 (y 60..360), drawn at x origin 0.
GLYPHS = {
    "F": "M-4,60 L212,60 L216,138 L204,138 L182,96 L86,92 L86,190 L150,188 "
         "L162,168 L172,168 L172,238 L162,238 L150,216 L86,214 L86,330 "
         "L124,352 L128,360 L-8,360 L-4,352 L30,330 L30,92 L-4,68 Z",
    "E": "M-4,60 L212,60 L216,138 L204,138 L182,96 L86,92 L86,190 L150,188 "
         "L162,168 L172,168 L172,238 L162,238 L150,216 L86,214 L86,326 "
         "L196,326 L222,272 L234,272 L228,360 L-8,360 L-4,352 L30,330 "
         "L30,92 L-4,68 Z",
    "R": "M-4,60 L170,60 C226,60 252,96 252,140 C252,180 228,204 188,214 "
         "L250,338 L270,352 L272,360 L180,360 L182,352 L196,344 L140,222 "
         "L86,222 L86,330 L120,352 L124,360 L-8,360 L-4,352 L30,330 "
         "L30,92 L-4,68 Z "
         "M86,90 L150,90 C182,90 194,112 194,140 C194,172 180,194 150,194 "
         "L86,194 Z",
    "A": "M118,50 L152,50 L252,330 L272,350 L274,360 L160,360 L162,352 "
         "L186,342 L166,284 L74,284 L56,336 L86,352 L88,360 L-12,360 "
         "L-10,352 L14,336 Z "
         "M134,122 L158,256 L84,256 Z",
    "L": "M-4,60 L122,60 L118,68 L86,92 L86,326 L192,326 L222,266 L234,266 "
         "L228,360 L-8,360 L-4,352 L30,330 L30,92 L-4,68 Z",
}
LAYOUT = [("F", 120), ("E", 368), ("R", 628), ("A", 926), ("L", 1224)]

# layout name -> (canvas W, H, letters, scale, translate x, translate y)
# The letters are drawn in wordmark coordinates and placed on the canvas
# with translate(tx, ty) scale(s).
LAYOUTS = {
    "wide": (1500, 520, LAYOUT, 1.0, 0.0, 0.0),
    "square": (1024, 1024, LAYOUT, 0.7, 512 - 0.7 * 785,
               530 - 0.7 * (DY + 210)),
    "icon": (1024, 1024, [("F", 0)], 2.0, 512 - 2.0 * 104,
             530 - 2.0 * (DY + 210)),
}

# Hammer cracks: short dark slashes across faces (letter, path in glyph coords)
CRACKS = [
    ("F", "M40,250 L70,226 L78,232"),
    ("E", "M150,326 L120,350"),
    ("R", "M210,110 L232,150 L226,170"),
    ("A", "M200,200 L178,236 L188,252"),
    ("L", "M40,140 L64,112"),
]


def glyph_paths(layout):
    out = []
    for ch, x in layout:
        out.append(f'<path transform="translate({x},{DY})" d="{GLYPHS[ch]}"/>')
    return "\n      ".join(out)


def cracks(layout):
    xs = dict(layout)
    return "\n    ".join(
        f'<path transform="translate({xs[ch]},{DY})" d="{d}"/>'
        for ch, d in CRACKS if ch in xs)


def streaks(rng, W, yt, h):
    """Diagonal sparks over the upper right; yt is the cap line, h the
    cap height, both in canvas coordinates."""
    out = []
    for _ in range(34):
        x = rng.uniform(0.25 * W, 0.83 * W)
        y = rng.uniform(max(0, yt - 0.45 * h), yt + 0.05 * h)
        L = rng.uniform(0.4 * h, 1.4 * h)
        ang = math.radians(58 + rng.uniform(-3, 3))
        x1, y1 = x + L * math.cos(ang), y + L * math.sin(ang)
        hot = rng.random() < 0.25
        col = "#E8641E" if hot else "#6B6F76"
        op = rng.uniform(0.15, 0.45)
        sw = rng.uniform(1.0, 3.5)
        out.append(
            f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x1:.0f}" y2="{y1:.0f}" '
            f'stroke="{col}" stroke-width="{sw:.1f}" opacity="{op:.2f}" '
            f'stroke-linecap="round"/>')
    return "\n    ".join(out)


def embers(rng, W, H, yb, h):
    out = []
    for _ in range(140):
        # cluster near the lower edge of the wordmark and the flanks
        x = rng.gauss(W / 2, 0.29 * W)
        y = rng.gauss(yb - 0.1 * h, 0.3 * h)
        if not (20 < x < W - 20 and 10 < y < H - 10):
            continue
        r = rng.choice([1.2, 1.5, 2, 2, 2.6, 3.4])
        col = rng.choice(["#FF8A2A", "#FF5A12", "#FFB347", "#F0440B"])
        op = rng.uniform(0.5, 1.0)
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" '
                   f'fill="{col}" opacity="{op:.2f}"/>')
    return "\n    ".join(out)


def main():
    out = sys.argv[1]
    bg = "none"
    if "--bg" in sys.argv:
        bg = sys.argv[sys.argv.index("--bg") + 1]
    layout = "wide"
    if "--layout" in sys.argv:
        layout = sys.argv[sys.argv.index("--layout") + 1]
    W, H, letters, sc, tx, ty = LAYOUTS[layout]
    place = f"translate({tx:.1f} {ty:.1f}) scale({sc})"
    yt, yb = ty + sc * (DY + 60), ty + sc * (DY + 360)
    h = yb - yt
    hx, hy, hrx, hry = W / 2, yb + 0.07 * h, 0.41 * W, 0.67 * h
    rng = random.Random(7)

    background = {
        "none": "",
        "white": f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        "dark": f'<rect width="{W}" height="{H}" fill="#14110F"/>',
    }[bg]

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">FERAL</title>
  <desc id="desc">Chiseled steel FERAL wordmark with forge-hot orange lower edges, embers and diagonal sparks.</desc>
  <defs>
    <linearGradient id="steel" x1="0" y1="60" x2="0" y2="360" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#D2D6DA"/>
      <stop offset="0.30" stop-color="#959BA3"/>
      <stop offset="0.55" stop-color="#6E747C"/>
      <stop offset="0.74" stop-color="#4C4F54"/>
      <stop offset="0.86" stop-color="#8A3A14"/>
      <stop offset="0.95" stop-color="#E8581A"/>
      <stop offset="1" stop-color="#FFB050"/>
    </linearGradient>
    <radialGradient id="heat" cx="{hx:.1f}" cy="{hy:.1f}" r="{hrx:.1f}" gradientUnits="userSpaceOnUse" gradientTransform="translate(0 {hy:.1f}) scale(1 {hry / hrx:.3f}) translate(0 {-hy:.1f})">
      <stop offset="0" stop-color="#FF6A1A" stop-opacity="0.55"/>
      <stop offset="0.5" stop-color="#F0440B" stop-opacity="0.18"/>
      <stop offset="1" stop-color="#F0440B" stop-opacity="0"/>
    </radialGradient>
    <!-- hammered grain + chisel bevel lighting -->
    <filter id="forged" x="-5%" y="-10%" width="110%" height="120%">
      <feTurbulence type="fractalNoise" baseFrequency="0.9 0.25" numOctaves="3" seed="4" result="noise"/>
      <feColorMatrix in="noise" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -0.9 0.42" result="grain"/>
      <feComposite in="grain" in2="SourceAlpha" operator="in" result="grainIn"/>
      <feGaussianBlur in="SourceAlpha" stdDeviation="3.2" result="blur"/>
      <feSpecularLighting in="blur" surfaceScale="9" specularConstant="1.3" specularExponent="18" lighting-color="#FFF4E6" result="spec">
        <feDistantLight azimuth="235" elevation="38"/>
      </feSpecularLighting>
      <feComposite in="spec" in2="SourceAlpha" operator="in" result="specIn"/>
      <feComposite in="SourceGraphic" in2="specIn" operator="arithmetic" k1="0" k2="1" k3="0.38" k4="0" result="lit"/>
      <feComposite in="grainIn" in2="lit" operator="over"/>
    </filter>
    <filter id="glow" x="-10%" y="-30%" width="120%" height="160%">
      <feGaussianBlur stdDeviation="9"/>
    </filter>
    <filter id="emberGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="1.6" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  {background}
  <ellipse cx="{hx:.1f}" cy="{hy:.1f}" rx="{hrx:.1f}" ry="{hry:.1f}" fill="url(#heat)"/>
  <g id="streaks">
    {streaks(rng, W, yt, h)}
  </g>
  <g id="letters" transform="{place}">
  <g id="hot-glow" fill="#FF5A12" fill-rule="evenodd" opacity="0.75" filter="url(#glow)" transform="translate(0 8)">
      {glyph_paths(letters)}
  </g>
  <g id="shadow" fill="#1B1714" fill-rule="evenodd" transform="translate(5 6)" opacity="0.55">
      {glyph_paths(letters)}
  </g>
  <g id="wordmark" fill="url(#steel)" fill-rule="evenodd" stroke="#24201D" stroke-width="2.5" stroke-linejoin="miter" filter="url(#forged)">
      {glyph_paths(letters)}
  </g>
  <g id="cracks" fill="none" stroke="#2A1A12" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" opacity="0.85">
    {cracks(letters)}
  </g>
  </g>
  <g id="embers" filter="url(#emberGlow)">
    {embers(rng, W, H, yb, h)}
  </g>
</svg>
'''
    with open(out, "w") as f:
        f.write(svg)


if __name__ == "__main__":
    main()
