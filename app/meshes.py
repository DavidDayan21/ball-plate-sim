"""Small mesh library for the parts of the robot shown in the window (app/parts.py). Without Qt.

A Mesh is vertices, normals and triangles, in the coordinates of the body it belongs to; the normals are given (sharp
or smooth edges, as each shape needs). Built with NumPy, written in MuJoCo's binary mesh format (MSH).
"""
import math
import struct

import numpy as np


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


class Mesh:
    def __init__(self, vertices=None, normals=None, faces=None):
        self.v = np.zeros((0, 3)) if vertices is None else np.asarray(vertices, float).reshape(-1, 3)
        self.n = np.zeros((0, 3)) if normals is None else unit(np.asarray(normals, float).reshape(-1, 3))
        self.f = np.zeros((0, 3), int) if faces is None else np.asarray(faces, int).reshape(-1, 3)

    @staticmethod
    def join(meshes) -> "Mesh":
        meshes = [m for m in meshes if len(m.f)]
        offsets = np.cumsum([0] + [len(m.v) for m in meshes[:-1]])
        if not meshes:
            return Mesh()
        return Mesh(np.concatenate([m.v for m in meshes]), np.concatenate([m.n for m in meshes]),
                    np.concatenate([m.f + k for m, k in zip(meshes, offsets)]))

    def oriented(self) -> "Mesh":
        """Triangles turned so that their own normal agrees with the normals of their vertices."""
        a, b, c = (self.v[self.f[:, k]] for k in range(3))
        agree = (np.cross(b - a, c - a) * (self.n[self.f].sum(axis=1))).sum(axis=1)
        flip = agree < 0
        self.f[flip] = self.f[flip][:, [0, 2, 1]]
        return self

    def msh(self) -> bytes:
        """MuJoCo's binary mesh format: counts (vertices, normals, texture coordinates, faces), then the arrays."""
        return (struct.pack("4i", len(self.v), len(self.n), 0, len(self.f)) + self.v.astype(np.float32).tobytes()
                + self.n.astype(np.float32).tobytes() + self.f.astype(np.int32).tobytes())


class Frame:
    """Origin o and axes e1, e2, e3 (the axes need not be right-handed: the triangles are oriented afterwards)."""

    def __init__(self, o, e1, e2, e3):
        self.o = np.asarray(o, float)
        self.R = np.array([e1, e2, e3], float)

    @property
    def e1(self):
        return self.R[0]

    @property
    def e2(self):
        return self.R[1]

    @property
    def e3(self):
        return self.R[2]

    def p(self, a, b, c):
        return self.o + np.array([a, b, c], float) @ self.R

    def points(self, local):
        return self.o + np.asarray(local, float) @ self.R

    def dirs(self, local):
        return np.asarray(local, float) @ self.R

    def at(self, a, b, c) -> "Frame":
        """The same axes, the origin moved to the point (a, b, c) of this frame."""
        return Frame(self.p(a, b, c), *self.R)


def _quads(n_rings: int, n: int, closed: bool = True):
    """Triangles between consecutive rings of n vertices (ring k: vertices k·n to k·n + n - 1)."""
    k = np.arange(n if closed else n - 1)
    nxt = (k + 1) % n
    faces = []
    for r in range(n_rings - 1):
        a, b = r * n + k, r * n + nxt
        faces += [np.stack([a, b, a + n], 1), np.stack([b, b + n, a + n], 1)]
    return np.concatenate(faces)


def lathe(fr: Frame, profile, n=32, smooth=False) -> Mesh:
    """Surface of revolution about e3. profile: [(r, z), ...], going up the outside of the part (r ≥ 0). smooth:
    normals averaged at the inner points of the profile (round shapes); otherwise sharp edges at every point."""
    prof = np.asarray(profile, float)
    d = np.diff(prof, axis=0)
    seg = unit(np.stack([d[:, 1], -d[:, 0]], axis=1))          # (radial, axial) normal of each segment
    t = 2 * math.pi * np.arange(n) / n
    cos, sin = np.cos(t), np.sin(t)
    meshes = []
    for j in range(len(d)):
        (r0, z0), (r1, z1) = prof[j], prof[j + 1]
        if r0 < 1e-12 and r1 < 1e-12:
            continue
        n0 = n1 = seg[j]
        if smooth:
            if j > 0:
                n0 = unit(seg[j - 1] + seg[j])
            if j < len(d) - 1:
                n1 = unit(seg[j] + seg[j + 1])
        ring = lambda r, z: np.stack([r * cos, r * sin, np.full(n, z)], 1)
        v = np.concatenate([ring(r0, z0), ring(r1, z1)])
        nn = np.concatenate([np.stack([n0[0] * cos, n0[0] * sin, np.full(n, n0[1])], 1),
                             np.stack([n1[0] * cos, n1[0] * sin, np.full(n, n1[1])], 1)])
        meshes.append(Mesh(fr.points(v), fr.dirs(nn), _quads(2, n)))
    return Mesh.join(meshes).oriented()


def disc(fr: Frame, r_out, z, up=True, r_in=0.0, n=32) -> Mesh:
    """Flat ring (or disc) at height z, normal ±e3."""
    t = 2 * math.pi * np.arange(n) / n
    ring = lambda r: np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], 1)
    v = np.concatenate([ring(r_in), ring(r_out)])
    normal = fr.dirs([0.0, 0.0, 1.0 if up else -1.0])
    return Mesh(fr.points(v), np.tile(normal, (2 * n, 1)), _quads(2, n)).oriented()


def cylinder(fr: Frame, r, z0, z1, n=32, r_in=0.0) -> Mesh:
    """Closed cylinder along e3 between z0 and z1, or a tube if r_in > 0."""
    z0, z1 = min(z0, z1), max(z0, z1)
    parts = [lathe(fr, [(r, z0), (r, z1)], n), disc(fr, r, z1, True, r_in, n), disc(fr, r, z0, False, r_in, n)]
    if r_in > 0:
        parts.append(lathe(fr, [(r_in, z1), (r_in, z0)], n))
    return Mesh.join(parts).oriented()


def prism(fr: Frame, poly, z0, z1, smooth=False) -> Mesh:
    """Convex polygon [(x, y), ...] (counterclockwise) in the plane (e1, e2), extruded along e3 from z0 to z1.
    smooth: normals of the sides averaged at the corners (round outlines)."""
    z0, z1 = min(z0, z1), max(z0, z1)
    p = np.asarray(poly, float)
    k = len(p)
    c = p.mean(axis=0)
    caps = []
    for z, nz in ((z1, 1.0), (z0, -1.0)):
        v = np.concatenate([[[c[0], c[1], z]], np.column_stack([p, np.full(k, z)])])
        f = np.stack([np.zeros(k, int), 1 + np.arange(k), 1 + (np.arange(k) + 1) % k], 1)
        caps.append(Mesh(fr.points(v), np.tile(fr.dirs([0, 0, nz]), (k + 1, 1)), f))
    q = np.roll(p, -1, axis=0)
    edge = unit(np.column_stack([q[:, 1] - p[:, 1], p[:, 0] - q[:, 0]]))
    n0 = n1 = edge
    if smooth:
        n0, n1 = unit(np.roll(edge, 1, axis=0) + edge), unit(edge + np.roll(edge, -1, axis=0))
    v = np.stack([np.column_stack([p, np.full(k, z0)]), np.column_stack([q, np.full(k, z0)]),
                  np.column_stack([q, np.full(k, z1)]), np.column_stack([p, np.full(k, z1)])], 1).reshape(-1, 3)
    nn = np.stack([n0, n1, n1, n0], 1).reshape(-1, 2)
    nn = np.column_stack([nn, np.zeros(len(nn))])
    base = 4 * np.arange(k)
    f = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)])
    return Mesh.join(caps + [Mesh(fr.points(v), fr.dirs(nn), f)]).oriented()


def box(fr: Frame, x0, x1, y0, y1, z0, z1) -> Mesh:
    return prism(fr, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z0, z1)


def chamfered_square(half, chamfer):
    h, c = half, chamfer
    return [(-h + c, -h), (h - c, -h), (h, -h + c), (h, h - c), (h - c, h), (-h + c, h), (-h, h - c), (-h, -h + c)]


def hexagon(r_corner, a0=0.0):
    return [(r_corner * math.cos(a0 + k * math.pi / 3), r_corner * math.sin(a0 + k * math.pi / 3)) for k in range(6)]


def stadium(r0, r1, length, n=16):
    """Outline of a link: circle r0 at (0, 0), circle r1 at (length, 0), joined by their outer tangents."""
    phi = math.asin((r0 - r1) / length)                        # the tangents are tilted by phi
    big = [(r0 * math.cos(t), r0 * math.sin(t)) for t in np.linspace(math.pi / 2 - phi, 1.5 * math.pi + phi, n + 1)]
    small = [(length + r1 * math.cos(t), r1 * math.sin(t))
             for t in np.linspace(-math.pi / 2 + phi, math.pi / 2 - phi, n + 1)]
    return big + small


def rounded_rectangle(half_x, half_y, r, n=4):
    """Outline centered on (0, 0), corners of radius r."""
    pts = []
    for cx, cy, a0 in ((half_x - r, half_y - r, 0), (-half_x + r, half_y - r, 90), (-half_x + r, -half_y + r, 180),
                       (half_x - r, -half_y + r, 270)):
        pts += [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
                for a in np.linspace(a0, a0 + 90, n + 1)]
    return pts


def smooth_path(points, per=8):
    """Catmull-Rom curve through the points."""
    p = np.asarray(points, float)
    p = np.concatenate([p[:1], p, p[-1:]])
    t = np.arange(per)[:, None] / per
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2:-1])
    return np.concatenate(out)


def tube(points, r, n=10, caps=True) -> Mesh:
    """Round tube along a polyline (cables): frames carried along the curve, smooth normals."""
    pts = np.asarray(points, float)
    tang = unit(np.gradient(pts, axis=0))
    x = unit(np.cross(tang[0], [0.0, 0.0, 1.0] if abs(tang[0][2]) < 0.9 else [1.0, 0.0, 0.0]))
    xs = []
    for t in tang:
        x = unit(x - (x @ t) * t)
        xs.append(x)
    xs = np.array(xs)
    ys = np.cross(tang, xs)
    a = 2 * math.pi * np.arange(n) / n
    d = np.cos(a)[None, :, None] * xs[:, None, :] + np.sin(a)[None, :, None] * ys[:, None, :]
    parts = [Mesh((pts[:, None, :] + r * d).reshape(-1, 3), d.reshape(-1, 3), _quads(len(pts), n))]
    if caps:
        for i, up in ((0, False), (-1, True)):
            parts.append(disc(Frame(pts[i], xs[i], ys[i], tang[i]), r, 0.0, up, 0.0, n))
    return Mesh.join(parts).oriented()


def ribbon(points, half_width, half_thickness) -> Mesh:
    """Flat cable along a path in a plane x = constant, its width along x."""
    p = np.asarray(points, float)
    x = np.array([1.0, 0.0, 0.0])
    parts = []
    for a, b in zip(p[:-1], p[1:]):
        nrm = np.cross(unit(b - a), x)
        for s in (1, -1):
            off = s * half_thickness * nrm
            v = [a - half_width * x + off, a + half_width * x + off, b + half_width * x + off, b - half_width * x + off]
            parts.append(Mesh(v, [s * nrm] * 4, [(0, 1, 2), (0, 2, 3)]))
    return Mesh.join(parts).oriented()


# --- Text: rasterized with the DejaVu font of Matplotlib, cut into rectangles -----------------------------------
def text_runs(text: str, height: float, px: int = 48):
    """Rectangles (x0, x1, y0, y1) in meters covering the text, centered on (0, 0); `height`: that of a capital H."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    font = FT2Font(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
                                         fallback_to_default=False))

    def bitmap(s):
        font.set_size(px, 72)
        font.set_text(s, 0.0)
        font.draw_glyphs_to_bitmap(antialiased=True)
        return np.asarray(font.get_image()) > 127
    scale = height / bitmap("H").any(axis=1).sum()            # meters per pixel
    a = bitmap(text)
    h, w = a.shape
    runs = []
    for r in range(h):
        row = np.concatenate([[False], a[r], [False]]).astype(int)
        edges = np.flatnonzero(np.diff(row))
        for c0, c1 in zip(edges[::2], edges[1::2]):
            runs.append(((c0 - w / 2) * scale, (c1 - w / 2) * scale, (h / 2 - r - 1) * scale, (h / 2 - r) * scale))
    return runs


def text_flat(fr: Frame, text: str, height: float, z: float = 0.0) -> Mesh:
    """Text in the plane (e1, e2) of fr, at height z, read from the side of +e3 (engraved or printed)."""
    runs = np.array(text_runs(text, height))
    x0, x1, y0, y1 = runs.T
    corners = np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1), np.stack([x0, y1], 1)], 1)
    v = np.concatenate([corners.reshape(-1, 2), np.full((4 * len(runs), 1), z)], axis=1)
    base = 4 * np.arange(len(runs))
    f = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)])
    return Mesh(fr.points(v), np.tile(fr.dirs([0, 0, 1]), (len(v), 1)), f).oriented()


def text_on_sphere(radius: float, direction, up, text: str, height: float, lift: float = 6e-5) -> Mesh:
    """Text printed on a sphere centered on the origin, centered on `direction`, its letters upright along `up`."""
    d = unit(direction)
    e1 = unit(np.cross(up, d))
    e2 = np.cross(d, e1)
    runs = np.array(text_runs(text, height))
    x0, x1, y0, y1 = runs.T
    corners = np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1), np.stack([x0, y1], 1)], 1)
    q = unit(radius * d + corners[..., :1] * e1 + corners[..., 1:] * e2).reshape(-1, 3)
    base = 4 * np.arange(len(runs))
    f = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1)])
    return Mesh((radius + lift) * q, q, f).oriented()
