"""The figures that come from the numbers: where the angle falls and what it does as the machine grows."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .figure import CREAM, BLACK, OCHRE, TEAL, TERRACOTTA, svg_frame, line, text
from .summary import family, independent, real_machine

FAMILIES = [("chatbench", "ChatBench", "exam questions"), ("alphanli", "αNLI", "the most plausible explanation"),
            ("haiid-text", "HAIID, text", "sarcasm, census"), ("haiid-images", "HAIID, images", "art, cities, skin")]


def _map_x(v, a, b, A, B):
    return A + (v - a) / (b - a) * (B - A)


def _fmt_num(v, dec=0):
    return f"{v:.{dec}f}"


def _dot(x, y, r, color, filled):
    if filled:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r - 0.75}" fill="{CREAM}" stroke="{color}" stroke-width="1.5"/>'


def _columns(points, step):
    """A stack of dots for each interval of step degrees: the dot does not move, it stacks."""
    stacks = defaultdict(list)
    for p in sorted(points, key=lambda p: p["phi"]):
        stacks[round(p["phi"] / step) * step].append(p)
    return stacks


def where_it_falls(map_data, source, vision=None) -> str:
    """One dot per comparison, per data family, on the same scale of degrees."""
    A, B, a, b = 200.0, 836.0, 30.0, 110.0
    step, r, jump = 1.5, 5.0, 11.0
    conf = [c for c in map_data["comparisons"] if "error" not in c and "excluded" not in c and independent(c) and real_machine(c)]
    bands = []
    for name, title, subtitle in FAMILIES:
        pp = [{"phi": c["phi"], "below": c["phi_ci"][1] < 90, "source": False} for c in conf if family(c) == name]
        if name == "chatbench":
            pp += [{"phi": c["phi"], "below": c["phi_ci"][1] < 90, "source": True}
                   for c in source["comparisons"] if c.get("subject") == "all" and "phi_ci" in c]
        ref = None
        if name == "chatbench":
            ref = source["person_person"][0]["phi"]
        elif name == "alphanli":
            ref = math.degrees(math.acos(map_data["person_person_alphanli"]["rho"]))
        bands.append((title, subtitle, pp, ref))
    if vision:
        bands.append(("Recognizing objects", "one dot per network, median over conditions", vision, None))
    body = []
    y = 44.0
    grid, under_grid = [], []
    for title, subtitle, pp, ref in bands:
        stacks = _columns(pp, step)
        height = max(len(v) for v in stacks.values())
        h = height * jump + 30
        base = y + h - 14
        body += [text(24, base - h + 34, title, 14, 700), text(24, base - h + 52, subtitle, 12)]
        body.append(line(A, base + 6, B, base + 6, OCHRE, 0.8))
        for c, v in stacks.items():
            for i, p in enumerate(v):
                body.append(_dot(_map_x(c, a, b, A, B), base - i * jump, r, OCHRE if p.get("network") else TERRACOTTA if p["source"] else TEAL, p["below"]))
        if ref is not None:
            xr = _map_x(ref, a, b, A, B)
            body.append(f'<polygon points="{xr:.1f},{base - 2:.1f} {xr + 6:.1f},{base + 6:.1f} {xr:.1f},{base + 14:.1f} {xr - 6:.1f},{base + 6:.1f}" fill="{OCHRE}" stroke="{BLACK}" stroke-width="1"/>')
        y = base + 28
    end = y - 12
    for v in range(30, 111, 10):
        xv = _map_x(v, a, b, A, B)
        under_grid.append(line(xv, 40, xv, end, OCHRE, 0.6, "1 5"))
        under_grid.append(text(xv, end + 18, f"{v}°", 12, None, "middle"))
    x90 = _map_x(90, a, b, A, B)
    under_grid.append(line(x90, 26, x90, end, BLACK, 1.6, "5 4"))
    under_grid.append(text(x90, 20, "90°, the hypothesis", 13, 700, "middle"))
    leg = end + 44
    legend = [_dot(30, leg - 4, r, TEAL, True), text(42, leg, "interval entirely below 90°", 12),
              _dot(214, leg - 4, r, TEAL, False), text(226, leg, "includes 90°", 12),
              _dot(350, leg - 4, r, TERRACOTTA, True), text(362, leg, "GPT-4o, Llama (alone and in chat)", 12),
              f'<polygon points="620,{leg - 10:.1f} 626,{leg - 4:.1f} 620,{leg + 2:.1f} 614,{leg - 4:.1f}" fill="{OCHRE}" stroke="{BLACK}" stroke-width="1"/>',
              text(632, leg, "two people between them", 12),
              _dot(30, leg + 18, r, OCHRE, True), text(42, leg + 22, "one vision network (median over its conditions)", 12)]
    n = sum(len(b[2]) for b in bands)
    aria = (f"Dot plot: {n} comparisons between people and machines, split by data family, on the angle scale from 30 to 110 degrees. "
            "Almost all the dots are to the left of the 90 degree line: the angle is not a right angle, and it changes with the task.")
    return svg_frame(860, int(leg + 40), aria, under_grid + body + legend)


def _series(rows, xs, ys, color, filled, dash):
    pts = [(xs(r["dm"]), ys(r)) for r in rows]
    out = [line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color, 2, dash) for i in range(len(pts) - 1)]
    out += [_dot(x, y, 5, color, filled) for x, y in pts]
    return out


def scale(h7) -> str:
    """Two small multiples: the angle and the person's part, against the machine's sensitivity."""
    panels = [("The angle between person and machine", lambda r: r["phi"], 50, 95, [50, 60, 70, 80, 90], "{}°", 70, 400),
              ("The person's part of the ideal, cos θ", lambda r: r["cos_theta"], 0.3, 1.0, [0.4, 0.6, 0.8, 1.0], None, 490, 820)]
    body = []
    top, bottom, a, b = 60.0, 300.0, 0.4, 3.0
    for title, val, y0, y1, ticks, fmt, A, B in panels:
        body.append(text(A - 46, 30, title, 14, 700))
        for t in ticks:
            yt = _map_x(t, y0, y1, bottom, top)
            body.append(line(A, yt, B, yt, OCHRE, 0.6, "1 5"))
            body.append(text(A - 8, yt + 4, fmt.format(t) if fmt else _fmt_num(t, 1), 12, None, "end"))
        for t in (0.5, 1, 1.5, 2, 2.5, 3):
            xt = _map_x(t, a, b, A, B)
            body.append(line(xt, bottom, xt, bottom + 5, BLACK, 1))
            body.append(text(xt, bottom + 20, _fmt_num(t, 1), 12, None, "middle"))
        body.append(line(A, bottom, B, bottom, BLACK, 1))
        body.append(text((A + B) / 2, bottom + 40, "machine sensitivity (d′)", 12, None, "middle"))
        if y1 == 95:
            y90 = _map_x(90, y0, y1, bottom, top)
            body.append(line(A, y90, B, y90, BLACK, 1.4, "5 4"))
        xs = lambda d, A=A, B=B: _map_x(d, a, b, A, B)
        ys = lambda r, val=val, y0=y0, y1=y1: _map_x(val(r), y0, y1, bottom, top)
        for key, col in (("chatbench", TEAL), ("alphanli", TERRACOTTA)):
            if f"{key}/reason" in h7:
                body += _series(h7[f"{key}/reason"]["rows"], xs, ys, col, False, "4 4")
            if f"{key}/letter" in h7:
                body += _series(h7[f"{key}/letter"]["rows"], xs, ys, col, True, None)
    leg = 372
    body += [_dot(70, leg - 4, 5, TEAL, True), text(82, leg, "ChatBench", 12), _dot(170, leg - 4, 5, TERRACOTTA, True), text(182, leg, "αNLI", 12),
             line(250, leg - 4, 284, leg - 4, BLACK, 2), text(292, leg, "immediate answer", 12),
             line(400, leg - 4, 434, leg - 4, BLACK, 2, "4 4"), text(442, leg, "with reasoning", 12)]
    aria = ("Two charts side by side, one line per data family, models ordered by sensitivity. On the left the angle: on αNLI it falls as the "
            "machine grows, on ChatBench it falls with the immediate answer and has no stable trend with reasoning. On the right the person's part of the ideal, which falls in all cases.")
    return svg_frame(860, 390, aria, body)


def write(summary: dict, output: Path) -> None:
    map_data = json.loads(Path("results/map.json").read_text())
    source = json.loads(Path("results/chatbench-source.json").read_text())
    vision = [{"phi": r["phi_median"], "below": True, "source": False, "network": True} for r in summary.get("vision", {}).get("networks", [])]
    (output / "where-it-falls.svg").write_text(where_it_falls(map_data, source, vision), encoding="utf-8")
    (output / "scale.svg").write_text(scale(summary["h7"]), encoding="utf-8")
