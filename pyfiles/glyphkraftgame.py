#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GlyphKraft a keyboard-native ASCII art studio + puzzle dojo.
  A curses terminal app where you draw with keys, sculpt symmetry, overlay ghost targets,
  and solve thinker puzzles by matching silhouettes, weaving constellations, or painting
  with constraints — all while the engine tracks score, strokes, time, and your moves.

Two big modes:
  [S] Studio      → Free drawing with tools: Brush • Line • Rect • Circle • Flood • Text
                    + 2-way & 4-way symmetry, kaleido mirroring, gradient brushes, undo/redo.
  [C] Challenge   → “Shadow Match” (paint to match a ghost), “Kaleido Targets”, “Ink Economy”.
                    Hit [Space] to check similarity and score.

Controls (highlights):
  Arrows / WASD   move cursor
  B               brush   (place current glyph)
  L               line    (toggle start/end; Bresenham)
  R               rect    (two corners)
  O               circle  (center then radius)
  F               flood fill (from cursor)
  T               text    (type; [Enter] to finish)
  [ / ]           cycle glyph palette
  ; / '           cycle gradient set
  G               symmetry: V/H toggle
  K               kaleidoscope: 4-way mirror
  U / Y           undo / redo
  C               clear canvas
  P               save PNG-ish text (ASCII) to file
  I               import text art from a file (plain .txt)
  H               help panel toggle
  M               switch mode (Studio/Challenge)
  1..3            choose challenge
  Space           check score (Challenge)
  Q               quit

No external deps, just standard library (curses).
"""

from __future__ import annotations
import curses, time, math, random, os, sys, textwrap
from collections import deque, namedtuple

# ──────────────────────────────────────────────────────────────────────────────
# Small data types
# ──────────────────────────────────────────────────────────────────────────────

Point = namedtuple("Point", ["x", "y"])

class Timer:
    """Tiny context manager for profiling small sections."""
    def __init__(self): self.t0 = 0; self.dt = 0
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *_): self.dt = (time.perf_counter() - self.t0) * 1000

# ──────────────────────────────────────────────────────────────────────────────
# Palettes, gradients, glyphs (you can add your own!)
# ──────────────────────────────────────────────────────────────────────────────

PALETTES = [
    list(" .:-=+*#%@"),
    list(" `'^\",;:-_."),
    list("░▒▓█"),
    list("·•●◦○"),
    list(".,oO0@#"),
    list("⟂|/\\—–=+"),
]

GRADIENTS = [
    list(" .:-=+*#%@"),
    list(" .oO0@#"),
    list("`^\".;:-_"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Canvas
# ──────────────────────────────────────────────────────────────────────────────

class Canvas:
    """A simple 2D char buffer with undo/redo and drawing ops."""
    def __init__(self, w: int, h: int, bg: str=" "):
        self.w, self.h, self.bg = w, h, bg
        self.buf = [[bg for _ in range(w)] for _ in range(h)]
        self.history: deque[list[tuple[int,int,str,str]]] = deque(maxlen=256)  # patches
        self.future: list[list[tuple[int,int,str,str]]] = []

    def in_bounds(self, p: Point) -> bool:
        return 0 <= p.x < self.w and 0 <= p.y < self.h

    def get(self, p: Point) -> str:
        if not self.in_bounds(p): return self.bg
        return self.buf[p.y][p.x]

    def set(self, p: Point, ch: str, record=True):
        if not self.in_bounds(p): return
        old = self.buf[p.y][p.x]
        if old == ch: return
        self.buf[p.y][p.x] = ch
        if record:
            self._ensure_frame()
            self.history[-1].append((p.x, p.y, old, ch))

    def _ensure_frame(self):
        if not self.history or self.history[-1] is None:
            self.history.append([])

    def frame(self):
        """Start an undo frame for grouping changes."""
        self._ensure_frame()
        self.future.clear()

    def commit(self):
        """Seal current frame if it has content; otherwise discard."""
        if self.history and self.history[-1] == []:
            self.history.pop()

    def undo(self):
        if not self.history: return
        patch = self.history.pop()
        for x, y, old, new in patch:
            self.buf[y][x] = old
        self.future.append(patch)

    def redo(self):
        if not self.future: return
        patch = self.future.pop()
        for x, y, old, new in patch:
            self.buf[y][x] = new
        self.history.append(patch)

    def clear(self, bg: str=None):
        bg = self.bg if bg is None else bg
        self.frame()
        for y in range(self.h):
            for x in range(self.w):
                self.set(Point(x,y), bg, record=True)
        self.commit()

    # ── drawing ops ───────────────────────────────────────────────────────────

    def line(self, a: Point, b: Point, ch: str):
        self.frame()
        for p in bresenham(a, b):
            self.set(p, ch)
        self.commit()

    def rect(self, a: Point, b: Point, ch: str, fill=False):
        x0,x1 = sorted([a.x, b.x]); y0,y1 = sorted([a.y, b.y])
        self.frame()
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                if fill or y in (y0, y1) or x in (x0, x1):
                    self.set(Point(x,y), ch)
        self.commit()

    def circle(self, c: Point, r: int, ch: str, fill=False):
        self.frame()
        for y in range(c.y-r, c.y+r+1):
            for x in range(c.x-r, c.x+r+1):
                if not self.in_bounds(Point(x,y)): continue
                dx, dy = x-c.x, y-c.y
                d = dx*dx + dy*dy
                if fill:
                    if d <= r*r: self.set(Point(x,y), ch)
                else:
                    # thin ring
                    if abs(math.sqrt(d) - r) < 0.5: self.set(Point(x,y), ch)
        self.commit()

    def fill(self, start: Point, ch: str):
        target = self.get(start)
        if target == ch: return
        self.frame()
        q = [start]
        seen = set()
        while q:
            p = q.pop()
            if p in seen: continue
            seen.add(p)
            if not self.in_bounds(p): continue
            if self.get(p) != target: continue
            self.set(p, ch)
            q.extend([Point(p.x+1,p.y), Point(p.x-1,p.y),
                      Point(p.x,p.y+1), Point(p.x,p.y-1)])
        self.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def bresenham(a: Point, b: Point):
    """Yield points on line AB (inclusive)."""
    x0, y0, x1, y1 = a.x, a.y, b.x, b.y
    dx, dy = abs(x1-x0), -abs(y1-y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        yield Point(x0,y0)
        if x0 == x1 and y0 == y1: break
        e2 = 2*err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy

# ──────────────────────────────────────────────────────────────────────────────
# Ghost targets (procedural)
# ──────────────────────────────────────────────────────────────────────────────

def ghost_circle(w,h):
    c = Canvas(w,h," ")
    r = min(w,h)//4
    c.circle(Point(w//2, h//2), r, "#", fill=True)
    return c

def ghost_tree(w,h):
    c = Canvas(w,h," ")
    # Simple L-system-ish branching
    cx, cy = w//2, h-2
    length = max(4, h//4)
    stack = [(cx, cy, -math.pi/2, length)]
    while stack:
        x,y,ang,len_ = stack.pop()
        if len_ <= 1: continue
        x2 = int(x + math.cos(ang)*len_)
        y2 = int(y + math.sin(ang)*len_)
        for p in bresenham(Point(x,y), Point(x2,y2)):
            if 0 < p.x < w-1 and 0 < p.y < h-1:
                c.set(p, "#", record=False)
        branch = len_ * (0.65 + random.random()*0.1)
        stack.append((x2,y2, ang + 0.5 + random.random()*0.2, branch))
        stack.append((x2,y2, ang - 0.5 - random.random()*0.2, branch))
    return c

def ghost_spiral(w,h):
    c = Canvas(w,h," ")
    cx, cy = w//2, h//2
    r = 1
    ang = 0.0
    while r < min(w,h)//2 - 2:
        x = int(cx + r * math.cos(ang))
        y = int(cy + r * math.sin(ang))
        if 0 < x < w-1 and 0 < y < h-1:
            c.set(Point(x,y), "#", record=False)
        ang += 0.15
        r += 0.1
    return c

GHOSTS = [ghost_circle, ghost_tree, ghost_spiral]

def similarity(a: Canvas, b: Canvas) -> float:
    """Jaccard-like score on foreground cells."""
    assert a.w == b.w and a.h == b.h
    A = set((x,y) for y in range(a.h) for x in range(a.w) if a.buf[y][x] != a.bg)
    B = set((x,y) for y in range(b.h) for x in range(b.w) if b.buf[y][x] != b.bg)
    if not A and not B: return 1.0
    return len(A & B) / len(A | B) if (A or B) else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# App (curses)
# ──────────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        self.h, self.w = stdscr.getmaxyx()
        # leave room for HUD lines
        self.ch = max(10, self.h - 4)
        self.cw = self.w
        self.canvas = Canvas(self.cw, self.ch)
        self.cursor = Point(self.cw//2, self.ch//2)
        self.mode = "STUDIO"  # or "CHALLENGE"
        self.tool = "BRUSH"
        self.glyph_palette_idx = 0
        self.glyph_idx = 4
        self.gradient_idx = 0
        self.sym_v = False
        self.sym_h = False
        self.kaleido = False
        self.line_anchor: Point|None = None
        self.rect_anchor: Point|None = None
        self.circle_center: Point|None = None
        self.brush_ch = PALETTES[self.glyph_palette_idx][self.glyph_idx]
        self.help = False
        # challenge state
        self.ghost: Canvas|None = None
        self.strokes = 0
        self.start_time = time.time()
        self.challenge_name = None

    # ── rendering ─────────────────────────────────────────────────────────────

    def draw(self):
        self.stdscr.erase()
        # ghost overlay (Challenge)
        if self.mode == "CHALLENGE" and self.ghost:
            for y in range(self.ch):
                row_chars = []
                for x in range(self.cw):
                    ghost_ch = self.ghost.buf[y][x]
                    if ghost_ch != " ":
                        row_chars.append("·")  # faint dot
                    else:
                        row_chars.append(" ")
                self.stdscr.addstr(y, 0, "".join(row_chars))

        # canvas
        for y in range(self.ch):
            row = "".join(self.canvas.buf[y])
            self.stdscr.addstr(y, 0, row)

        # cursor
        try:
            self.stdscr.addstr(self.cursor.y, self.cursor.x, "@", curses.A_REVERSE)
        except curses.error:
            pass

        # HUD
        hud1 = f"Mode:{self.mode:<9} Tool:{self.tool:<7} Glyph:'{self.brush_ch}'  Sym(V/H):{int(self.sym_v)}/{int(self.sym_h)}  Kaleido:{int(self.kaleido)}  Size:{self.cw}x{self.ch}"
        hud2 = f"[Arrows/WASD move]  B/L/R/O/F/T tools  [ / ] glyph  ;/' gradient  G=mirror  K=kaleido  U/Y undo/redo  C clear  P save  I import  M mode  H help  Q quit"
        self.stdscr.addstr(self.ch, 0, hud1[:self.w-1])
        self.stdscr.addstr(self.ch+1, 0, hud2[:self.w-1])

        if self.mode == "CHALLENGE":
            elapsed = int(time.time() - self.start_time)
            chal = self.challenge_name or "Shadow"
            self.stdscr.addstr(self.ch+2, 0, f"Challenge:{chal}  Strokes:{self.strokes}  Time:{elapsed}s  [Space]=score")

        if self.help:
            self.draw_help()

        self.stdscr.refresh()

    def draw_help(self):
        panel = textwrap.dedent("""
        TOOLS: B brush • L line • R rect • O circle • F flood • T text
        SYMMETRY: G toggles vertical/horizontal mirror; K toggles 4-way kaleidoscope.
        PALETTE: [ / ] to cycle glyph; ; / ' to cycle gradient set.
        CHALLENGES: 1 Shadow Match • 2 Kaleido Target • 3 Ink Economy.
        SPACE (in Challenge) to score against the ghost.
        SAVE/LOAD: P save to ascii file • I import .txt file (same size works best).
        """).strip("\n")
        lines = panel.splitlines()
        top = 2
        for i, ln in enumerate(lines):
            if top+i < self.ch:
                self.stdscr.addstr(top+i, 2, ln[:self.w-4], curses.A_DIM)

    # ── input + actions ───────────────────────────────────────────────────────

    def run(self):
        while True:
            self.draw()
            ch = self.stdscr.getch()
            if ch in (ord('q'), ord('Q')): break
            self.handle_key(ch)

    def handle_key(self, ch):
        # movement
        if ch in (curses.KEY_LEFT, ord('a'), ord('A')):  self.move(-1,0)
        elif ch in (curses.KEY_RIGHT, ord('d'), ord('D')): self.move(1,0)
        elif ch in (curses.KEY_UP, ord('w'), ord('W')):    self.move(0,-1)
        elif ch in (curses.KEY_DOWN, ord('s'), ord('S')):  self.move(0,1)

        # mode toggle
        elif ch in (ord('m'), ord('M')): self.toggle_mode()

        # glyph palette cycling
        elif ch == ord('['): self.prev_glyph()
        elif ch == ord(']'): self.next_glyph()
        elif ch == ord(';'): self.prev_gradient()
        elif ch == ord('\''): self.next_gradient()

        # tools
        elif ch in (ord('b'), ord('B')): self.use_brush()
        elif ch in (ord('l'), ord('L')): self.use_line()
        elif ch in (ord('r'), ord('R')): self.use_rect()
        elif ch in (ord('o'), ord('O')): self.use_circle()
        elif ch in (ord('f'), ord('F')): self.use_fill()
        elif ch in (ord('t'), ord('T')): self.use_text()

        # symmetry
        elif ch in (ord('g'), ord('G')): self.toggle_symmetry()
        elif ch in (ord('k'), ord('K')): self.kaleido = not self.kaleido

        # undo/redo
        elif ch in (ord('u'), ord('U')): self.canvas.undo()
        elif ch in (ord('y'), ord('Y')): self.canvas.redo()

        # canvas ops
        elif ch in (ord('c'), ord('C')): self.canvas.clear()
        elif ch in (ord('p'), ord('P')): self.save_ascii()
        elif ch in (ord('i'), ord('I')): self.load_ascii()

        # help
        elif ch in (ord('h'), ord('H')): self.help = not self.help

        # challenges
        elif ch == ord('1'): self.start_challenge("Shadow", ghost_fn=random.choice(GHOSTS))
        elif ch == ord('2'): self.start_challenge("Kaleido Target", ghost_fn=ghost_spiral, force_kaleido=True)
        elif ch == ord('3'): self.start_challenge("Ink Economy", ghost_fn=ghost_tree, ink_cap=200)
        elif ch == ord(' '): self.score_challenge()

    # ── movement and painting primitives ──────────────────────────────────────

    def move(self, dx, dy):
        x = max(0, min(self.cw-1, self.cursor.x + dx))
        y = max(0, min(self.ch-1, self.cursor.y + dy))
        self.cursor = Point(x,y)

    def paint(self, p: Point, ch: str):
        # symmetry transforms
        targets = [p]
        if self.sym_v:
            targets.append(Point(self.cw-1 - p.x, p.y))
        if self.sym_h:
            targets.append(Point(p.x, self.ch-1 - p.y))
        if self.sym_v and self.sym_h:
            targets.append(Point(self.cw-1 - p.x, self.ch-1 - p.y))
        if self.kaleido:
            cx, cy = self.cw//2, self.ch//2
            dx, dy = p.x - cx, p.y - cy
            targets.extend([
                Point(cx - dx, cy + dy),
                Point(cx + dx, cy - dy),
                Point(cx - dx, cy - dy),
            ])
        self.canvas.frame()
        for tp in targets:
            self.canvas.set(tp, ch)
        self.canvas.commit()
        if self.mode == "CHALLENGE":
            self.strokes += 1

    # ── tools ─────────────────────────────────────────────────────────────────

    def use_brush(self):
        self.tool = "BRUSH"
        ch = self.brush_ch
        self.paint(self.cursor, ch)

    def use_line(self):
        self.tool = "LINE"
        if self.line_anchor is None:
            self.line_anchor = self.cursor
        else:
            # draw using current gradient from start→end
            pts = list(bresenham(self.line_anchor, self.cursor))
            grad = GRADIENTS[self.gradient_idx]
            self.canvas.frame()
            for i, p in enumerate(pts):
                gch = grad[i % len(grad)]
                for tp in self.transform_targets(p):
                    self.canvas.set(tp, gch)
            self.canvas.commit()
            self.line_anchor = None
            if self.mode == "CHALLENGE":
                self.strokes += len(pts)

    def use_rect(self):
        self.tool = "RECT"
        if self.rect_anchor is None:
            self.rect_anchor = self.cursor
        else:
            a, b = self.rect_anchor, self.cursor
            for tp_a, tp_b in self.transform_pair(a, b):
                self.canvas.rect(tp_a, tp_b, self.brush_ch, fill=False)
            self.rect_anchor = None

    def use_circle(self):
        self.tool = "CIRCLE"
        if self.circle_center is None:
            self.circle_center = self.cursor
        else:
            r = int(round(dist(self.circle_center, self.cursor)))
            centers = self.transform_targets(self.circle_center)
            for cc in centers:
                self.canvas.circle(cc, r, self.brush_ch, fill=False)
            self.circle_center = None

    def use_fill(self):
        self.tool = "FILL"
        for tp in self.transform_targets(self.cursor):
            self.canvas.fill(tp, self.brush_ch)

    def use_text(self):
        self.tool = "TEXT"
        curses.curs_set(1)
        try:
            s = self.prompt("Type text (Enter to commit, Esc to cancel): ")
            if s is not None:
                self.canvas.frame()
                x,y = self.cursor.x, self.cursor.y
                for i,ch in enumerate(s):
                    p = Point(min(self.cw-1, x+i), y)
                    for tp in self.transform_targets(p):
                        self.canvas.set(tp, ch)
                self.canvas.commit()
        finally:
            curses.curs_set(0)

    # ── helpers for transforms ────────────────────────────────────────────────

    def transform_targets(self, p: Point):
        targets = [p]
        if self.sym_v:
            targets.append(Point(self.cw-1 - p.x, p.y))
        if self.sym_h:
            targets.append(Point(p.x, self.ch-1 - p.y))
        if self.sym_v and self.sym_h:
            targets.append(Point(self.cw-1 - p.x, self.ch-1 - p.y))
        if self.kaleido:
            cx, cy = self.cw//2, self.ch//2
            dx, dy = p.x - cx, p.y - cy
            targets.extend([
                Point(cx - dx, cy + dy),
                Point(cx + dx, cy - dy),
                Point(cx - dx, cy - dy),
            ])
        # deduplicate and in-bounds
        uniq = []
        seen = set()
        for tp in targets:
            if (tp.x,tp.y) in seen: continue
            seen.add((tp.x,tp.y))
            if 0 <= tp.x < self.cw and 0 <= tp.y < self.ch:
                uniq.append(tp)
        return uniq

    def transform_pair(self, a: Point, b: Point):
        """Yield mirrored pairs for segment-like shapes."""
        yield a, b
        if self.sym_v: yield Point(self.cw-1-a.x, a.y), Point(self.cw-1-b.x, b.y)
        if self.sym_h: yield Point(a.x, self.ch-1-a.y), Point(b.x, self.ch-1-b.y)
        if self.sym_v and self.sym_h:
            yield Point(self.cw-1-a.x, self.ch-1-a.y), Point(self.cw-1-b.x, self.ch-1-b.y)
        if self.kaleido:
            cx, cy = self.cw//2, self.ch//2
            dx1, dy1 = a.x - cx, a.y - cy
            dx2, dy2 = b.x - cx, b.y - cy
            yield Point(cx - dx1, cy + dy1), Point(cx - dx2, cy + dy2)
            yield Point(cx + dx1, cy - dy1), Point(cx + dx2, cy - dy2)
            yield Point(cx - dx1, cy - dy1), Point(cx - dx2, cy - dy2)

    # ── glyph + sym toggles ──────────────────────────────────────────────────

    def prev_glyph(self): self.cycle_glyph(-1)
    def next_glyph(self): self.cycle_glyph(+1)

    def cycle_glyph(self, d):
        pal = PALETTES[self.glyph_palette_idx]
        self.glyph_idx = (self.glyph_idx + d) % len(pal)
        self.brush_ch = pal[self.glyph_idx]

    def prev_gradient(self): self.cycle_gradient(-1)
    def next_gradient(self): self.cycle_gradient(+1)
    def cycle_gradient(self, d):
        self.gradient_idx = (self.gradient_idx + d) % len(GRADIENTS)

    def toggle_symmetry(self):
        # cycle: none → V → H → V+H → none
        if not self.sym_v and not self.sym_h:
            self.sym_v = True
        elif self.sym_v and not self.sym_h:
            self.sym_v = False; self.sym_h = True
        elif not self.sym_v and self.sym_h:
            self.sym_v = True; self.sym_h = True
        else:
            self.sym_v = self.sym_h = False

    # ── mode + challenges ────────────────────────────────────────────────────

    def toggle_mode(self):
        self.mode = "CHALLENGE" if self.mode == "STUDIO" else "STUDIO"
        if self.mode == "CHALLENGE":
            self.start_challenge("Shadow", ghost_fn=random.choice(GHOSTS))
        else:
            self.ghost = None
            self.challenge_name = None

    def start_challenge(self, name: str, ghost_fn, force_kaleido=False, ink_cap=None):
        self.mode = "CHALLENGE"
        self.canvas.clear()
        self.ghost = ghost_fn(self.cw, self.ch)
        self.strokes = 0
        self.start_time = time.time()
        self.challenge_name = name
        self.kaleido = force_kaleido
        self.ink_cap = ink_cap

    def score_challenge(self):
        if self.mode != "CHALLENGE" or not self.ghost: return
        score = similarity(self.canvas, self.ghost)
        elapsed = int(time.time() - self.start_time)
        msg = f"Similarity: {score*100:5.1f}%   Strokes:{self.strokes}   Time:{elapsed}s"
        if self.ink_cap:
            msg += f"   Ink cap:{self.ink_cap}  {'OK' if self.strokes<=self.ink_cap else 'EXCEEDED'}"
        self.message(msg)

    # ── persistence ──────────────────────────────────────────────────────────

    def save_ascii(self):
        ts = int(time.time())
        fn = f"glyph_{ts}.txt"
        try:
            with open(fn, "w", encoding="utf-8") as f:
                for y in range(self.ch):
                    f.write("".join(self.canvas.buf[y]) + "\n")
            self.message(f"Saved {fn}")
        except Exception as e:
            self.message(f"Save failed: {e}")

    def load_ascii(self):
        curses.curs_set(1)
        try:
            path = self.prompt("Import .txt path: ")
            if not path: return
            lines = open(path, "r", encoding="utf-8").read().splitlines()
            H = min(self.ch, len(lines))
            W = min(self.cw, max((len(ln) for ln in lines), default=0))
            self.canvas.frame()
            for y in range(H):
                for x in range(min(W, len(lines[y]))):
                    self.canvas.set(Point(x,y), lines[y][x])
            self.canvas.commit()
            self.message(f"Imported {path}")
        except Exception as e:
            self.message(f"Import failed: {e}")
        finally:
            curses.curs_set(0)

    # ── utility UI ───────────────────────────────────────────────────────────

    def message(self, s: str, ms=1600):
        self.stdscr.addstr(self.ch+3, 0, (" " * (self.w-1)))
        self.stdscr.addstr(self.ch+3, 0, s[:self.w-1])
        self.stdscr.refresh()
        curses.napms(ms)

    def prompt(self, s: str) -> str|None:
        self.stdscr.addstr(self.ch+3, 0, (" " * (self.w-1)))
        self.stdscr.addstr(self.ch+3, 0, s[:self.w-1])
        self.stdscr.refresh()
        curses.echo()
        curses.curs_set(1)
        try:
            resp = self.stdscr.getstr(self.ch+3, len(s), 200)
            if resp is None: return None
            return resp.decode("utf-8")
        finally:
            curses.noecho()
            curses.curs_set(0)

# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def dist(a: Point, b: Point) -> float:
    return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main(stdscr):
    app = App(stdscr)
    app.run()

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except curses.error as e:
        sys.stderr.write("This app needs a real terminal (curses). On Windows, install windows-curses.\n")
        raise
