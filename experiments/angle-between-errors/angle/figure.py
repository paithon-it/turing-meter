"""The figures of the article, as SVG with the fixed palette. The classes for the two themes are added
by a separate step when the figures are published: here only the fallback colors remain.

Usage, from the experiment folder:  python -m angle.figure
"""
from __future__ import annotations

import json
import math
from pathlib import Path

TERRACOTTA, TEAL, OCHRE, BLACK, CREAM = "#B5532C", "#2D5A5C", "#C9A961", "#1A1A1A", "#F8F5EE"
OUTPUT = Path("results/figure")


def text(x, y, s, size=13, weight=None, anchor="start", color=BLACK):
    p = f' font-weight="{weight}"' if weight else ""
    return f'<text font-family="sans-serif" x="{x:.1f}" y="{y:.1f}" font-size="{size}"{p} fill="{color}" text-anchor="{anchor}">{s}</text>'


def line(x1, y1, x2, y2, color=BLACK, w=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"{d}/>'


def arrow(x1, y1, x2, y2, color, w=4):
    a = math.atan2(y2 - y1, x2 - x1)
    l = 12.0
    bx, by = x2 - l * math.cos(a), y2 - l * math.sin(a)
    px, py = 6.5 * math.sin(a), -6.5 * math.cos(a)
    return (line(x1, y1, bx, by, color, w) +
            f'<polygon points="{x2:.1f},{y2:.1f} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="{color}"/>')


def svg_frame(l, h, aria, body):
    return (f'<svg viewBox="0 0 {l} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{aria}">\n'
            f'<rect width="{l}" height="{h}" fill="{CREAM}"/>\n' + "\n".join(body) + "\n</svg>\n")


def ideal_vector(dh, dm, phi):
    f = math.radians(phi)
    sx = dh
    sy = (dm - dh * math.cos(f)) / math.sin(f)
    return sx, sy


def geometry() -> str:
    """Three panels: ninety degrees, the measured angle, the useful angle."""
    dh, dm = 1.08, 2.34
    scale, base, body = 82.0, 300.0, []
    panels = [(90.0, "90 degrees, by hypothesis", "the Pythagorean law"), (75.0, "75 degrees, measured", "GPT-4o on ChatBench"),
              (math.degrees(math.acos(dh / dm)), "The useful angle, 62 degrees", "the person adds nothing")]
    for i, (phi, title, subtitle) in enumerate(panels):
        x0 = 44 + i * 276
        body += [text(x0 - 8, 32, title, 15, 700), text(x0 - 8, 52, subtitle, 13)]
        f = math.radians(phi)
        sx, sy = ideal_vector(dh, dm, phi)
        S = (x0 + sx * scale, base - sy * scale)
        H = (x0 + dh * scale, base)
        M = (x0 + dm * math.cos(f) * scale, base - dm * math.sin(f) * scale)
        body.append(line(x0, base, x0 + 2.75 * scale, base, TEAL, 1, "2 4"))
        body.append(line(x0, base, x0 + 2.75 * scale * math.cos(f), base - 2.75 * scale * math.sin(f), TERRACOTTA, 1, "2 4"))
        body.append(line(S[0], S[1], H[0], H[1], BLACK, 1.2, "5 4"))
        body.append(line(S[0], S[1], M[0], M[1], BLACK, 1.2, "5 4"))
        body.append(arrow(x0, base, S[0], S[1], BLACK, 4))
        body.append(arrow(x0, base, H[0], H[1], TEAL, 6))
        body.append(arrow(x0, base, M[0], M[1], TERRACOTTA, 6))
        norm = math.hypot(sx, sy)
        body.append(text(H[0], base + 22, "person 1.08", 13, 700, "end", TEAL))
        body.append(text(M[0] + 8, M[1] - 8, "machine 2.34", 13, 700, "start", TERRACOTTA))
        body.append(text(x0 + 6, base + 42, f"ideal pair {norm:.2f}", 13, 700))
    aria = ("Three diagrams with two axes starting from the same point, the person's horizontal and the machine's tilted. "
            "A dark arrow, the ideal pair, projects onto the two axes and its projections are the two sensitivities, 1.08 and 2.34. "
            "At 90 degrees the arrow is 2.58 long; at 75 degrees 2.39; at 62 degrees it coincides with the machine's axis, 2.34 long, "
            "and the person adds nothing more.")
    return svg_frame(860, 350, aria, body)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "geometry.svg").write_text(geometry(), encoding="utf-8")
    r = Path("results/summary.json")
    if r.exists():
        from . import figure_data
        figure_data.write(json.loads(r.read_text()), OUTPUT)
    print("ok")


if __name__ == "__main__":
    main()
