"""A gas grill run below its detent: an excitable medium of identical cells.

Each cell holds gas g and a flame size (both private) and heat f (its one
shared number). Burner discs feed gas at the knob's rate. A flame burns
size*k*g; it grows while that is at least q and shrinks away when it falls
below, so flames swell and die over many updates. An unlit cell holding more than
the lean limit lights when its neighbours' heat times its gas passes its
threshold (a strong flame lights thin gas, faint warmth only a rich pool).
Cells update at random times. The knob is in units of q, so the detent is at 1.

  python grill.py                          # play: knob slider, click to strike a match;
                                           # records to recordings/*.mp4, saved on Esc
  python grill.py --render out.mp4 --seconds 20 --knob 1.5:0,0.8:4
  python grill.py --measure --knob 0.8
"""
import argparse
import ctypes
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np

# Numba starts threads far faster with TBB (pip's tbb puts it in <venv>/lib, off the loader's path).
try:
    ctypes.CDLL(os.path.join(sys.prefix, 'lib', 'libtbb.so.12'), ctypes.RTLD_GLOBAL)
except OSError:
    pass
from numba import njit, prange  # noqa: E402

P = dict(
    p=0.5,        # chance a cell updates in a tick
    k=0.04,       # burn rate per unit gas, per update
    q=0.01,       # quench: a flame burning less than this shrinks away
    grow=0.05,    # rate a flame grows or shrinks, per update
    glean=3.0,    # lean limit: no less gas than this will light
    leak=0.001,   # unburnt gas drifting away, per update
    lam=0.97,     # heat kept per update
    mix=0.2,      # weight of a cell's own heat against its neighbours'
    theta=1.0,    # heat x gas needed to light (a sheet burning at q holds heat 1)
    sigma=0.001,  # noise on heat, per update
    var=0.1,      # fixed per-cell spread of theta
    spread=0.1,   # fixed per-burner spread of supply (hole sizes)
)


# ---------------------------------------------------------------- randomness

@njit(inline='always')
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(inline='always')
def _uniforms(seed, tick, i):
    """Two independent uniforms in [0,1) for this cell and tick, deterministic."""
    z = _mix(np.uint64(i) + np.uint64(tick) * np.uint64(0x9E3779B97F4A7C15) + seed)
    return (z >> np.uint64(40)) * (1.0 / 16777216.0), (z & np.uint64(0xFFFFFF)) * (1.0 / 16777216.0)


# ---------------------------------------------------------------- the cell

@njit(parallel=True, fastmath=True, cache=True)
def _run(f, fo, g, flame, supply, theta, ys, x0s, x1s, knob, seed, tick, ticks,
         p, k, q, grow, glean, leak, lam, mix, sigma):
    """Run the sheet for some ticks; returns the heat buffers in their new order."""
    H, W = f.shape
    inj = (1.0 - lam) / q          # a sheet burning at q settles at heat 1
    for t in range(tick, tick + ticks):
        for j in prange(len(ys)):                      # equal-length segments of the sheet
            y = ys[j]
            for x in range(x0s[j], x1s[j]):
                u, v = _uniforms(seed, t, y * W + x)
                if u >= p:
                    fo[y, x] = f[y, x]
                    continue
                # the shared number: neighbours' heat (the sheet never touches the border)
                h = 0.125 * (f[y - 1, x - 1] + f[y - 1, x] + f[y - 1, x + 1] + f[y, x - 1]
                             + f[y, x + 1] + f[y + 1, x - 1] + f[y + 1, x] + f[y + 1, x + 1])
                gc = g[y, x]
                fl = flame[y, x]
                if fl > 0.0:
                    fl += grow * ((1.0 if fl * k * gc >= q else 0.0) - fl)
                    if fl < 0.02:
                        fl = 0.0
                elif gc > glean and h * gc > theta[y, x]:
                    fl = 0.2                   # a new flame starts small
                flame[y, x] = fl
                b = fl * k * gc
                gc += knob * q * supply[y, x] - b - leak * gc
                g[y, x] = gc if gc > 0.0 else 0.0
                fn = lam * (mix * f[y, x] + (1.0 - mix) * h) + inj * b + sigma * (2.0 * v - 1.0)
                fo[y, x] = fn if fn > 0.0 else 0.0
        f, fo = fo, f
    return f, fo


def segments(mask, n=128):
    """The runs of True along each row, cut into pieces of at most n cells: (ys, x0s, x1s)."""
    H, W = mask.shape
    m = np.zeros((H, W + 2), np.int8)
    m[:, 1:-1] = mask
    d = np.diff(m, axis=1)
    ys, a = np.nonzero(d == 1)
    _, b = np.nonzero(d == -1)
    out = [(y, x, min(x + n, e)) for y, s, e in zip(ys, a, b) for x in range(s, e, n)]
    return tuple(np.array(c, np.int64) for c in zip(*out))


# ---------------------------------------------------------------- the world

def eight(W, H, n, fill):
    """n holes round each loop of a horizontal figure of 8 (a lemniscate), one shared at the crossing."""
    a = 0.46 * min(W, H / 0.72)                          # half-width; the 8 is 0.71 a high
    t = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 20001)
    d = 1 + np.sin(t) ** 2
    x, y = a * np.cos(t) / d, a * np.sin(t) * np.cos(t) / d
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    idx = np.searchsorted(arc, np.arange(2 * n) * arc[-1] / (2 * n))
    cs = np.delete(np.stack([W / 2 + x[idx], H / 2 + y[idx]], 1), n, 0)  # the crossing comes round twice
    gaps = np.hypot(*(cs[:, None] - cs[None]).transpose(2, 0, 1)) + np.eye(len(cs)) * 1e9
    return cs, fill * gaps.min() / 2


def honeycomb(W, H, n, fill):
    """The 8's holes, at the 8's spacing, packed hexagonally over the whole sheet."""
    r = eight(W, H, n, fill)[1]
    px = 2 * r / fill
    py = px * math.sqrt(3) / 2
    cs = np.array([((i + j % 2 / 2) * px, j * py)
                   for j in range(int((H - 2 * r) // py) + 1) for i in range(int(W // px) + 1)])
    cs = cs[cs[:, 0] <= W - 2 * r]                       # keep the holes that fit
    return cs + (np.array([W, H]) - cs.max(0) - cs.min(0)) / 2, r


def burners(W, H, n, fill, layout, rng):
    """Centres and radius of the circular holes over a W x H sheet."""
    if layout == 'eight':
        return eight(W, H, n, fill)
    if layout == 'honeycomb':
        return honeycomb(W, H, n, fill)
    pitch = math.sqrt(W * H / n)
    r = fill * pitch / 2
    if layout == 'random':
        cs, tries = [], 0
        while len(cs) < n and tries < 200 * n:
            c = rng.uniform((r, r), (W - r, H - r))
            tries += 1
            if all((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 > (0.8 * pitch) ** 2 for d in cs):
                cs.append(c)
        return np.array(cs), r
    nx, ny = max(1, round(W / pitch)), max(1, round(H / pitch))
    px, py = W / nx, H / ny
    cs = []
    for j in range(ny):
        off = px / 2 if layout == 'hex' and j % 2 else 0
        for i in range(nx - (layout == 'hex' and j % 2)):
            cs.append(((i + 0.5) * px + off, (j + 0.5) * py))
    return np.array(cs), r


class Grill:
    def __init__(self, W, H, n=10, fill=0.97, layout='eight', seed=1, **kw):
        self.P = {**P, **kw}
        self.W, self.H, self.seed = W, H, np.uint64(seed * 0x2545F4914F6CDD1D % 2**64)
        rng = np.random.default_rng(seed)
        self.centres, self.radius = burners(W, H, n, fill, layout, rng)
        self.supply = np.zeros((H, W), np.float32)
        r = self.radius
        for (cx, cy), s in zip(self.centres, 1 + self.P['spread'] * rng.standard_normal(len(self.centres))):
            x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 2)
            y0, y1 = max(0, int(cy - r)), min(H, int(cy + r) + 2)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            box = self.supply[y0:y1, x0:x1]
            box[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = s
        var = self.P['var']
        self.theta = (self.P['theta'] * (1 + var * rng.uniform(-1, 1, (H, W)))).astype(np.float32)
        # The sheet is the grill: the holes and the air around them out to where their heat has
        # died away (eight heat lengths). Beyond it is cold, like the edge.
        reach = 8 * math.sqrt(0.375 * (1 - self.P['mix']) / (1 - self.P['lam']))
        near = np.zeros((H, W), bool)
        R = r + reach
        for cx, cy in self.centres:
            x0, x1 = max(1, int(cx - R)), min(W - 1, int(cx + R) + 2)
            y0, y1 = max(1, int(cy - R)), min(H - 1, int(cy + R) + 2)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            near[y0:y1, x0:x1] |= (xx - cx) ** 2 + (yy - cy) ** 2 <= R * R
        self.sheet = near
        self.ys, self.x0s, self.x1s = segments(near)
        self.f = np.zeros((H, W), np.float32)
        self.fo = np.zeros_like(self.f)
        self.g = (self.supply > 0) * np.float32(3 * self.P['glean'])   # gas already on
        self.g = self.g.astype(np.float32)
        self.flame = np.zeros((H, W), np.float32)
        self.t = 0
        self.knob = 1.5

    def strike(self, x, y, r=None):
        """A match: heat at a point (the world, not a cell)."""
        r = r or max(3, int(self.radius))
        box = np.s_[max(0, y - r):y + r, max(0, x - r):x + r]
        self.f[box] += 3.0 * self.sheet[box]

    def run(self, ticks):
        P = self.P
        self.f, self.fo = _run(self.f, self.fo, self.g, self.flame, self.supply, self.theta,
                               self.ys, self.x0s, self.x1s, np.float32(self.knob), self.seed, self.t, ticks,
                               P['p'], P['k'], P['q'], P['grow'], P['glean'], P['leak'], P['lam'], P['mix'],
                               P['sigma'])
        self.t += ticks


# ---------------------------------------------------------------- light

BLUE = np.array([0.18, 0.38, 1.0], np.float32)     # CH 431 nm + C2 516 nm
SOOT = np.array([1.0, 0.52, 0.12], np.float32)     # incandescent soot, ~1900 K


@njit(parallel=True, fastmath=True, cache=True)
def _emit(flame, g, kq, glean, cb, cs, out):
    """Light from each cell: blue from burning, soot from burning rich; unlit cells are black."""
    H, W = g.shape
    for y in prange(H):
        for x in range(W):
            b = flame[y, x] * g[y, x] * kq             # burn rate, in units of q
            rich = (g[y, x] / glean - 1.5) / 3.0
            rich = min(max(rich, 0.0), 1.0)
            blue = b * (1.0 - rich)
            soot = 4.0 * b * rich
            for c in range(3):
                out[y, x, c] = blue * cb[c] + soot * cs[c]


@njit(parallel=True, fastmath=True, cache=True)
def _down(src, dst, s):
    h, w, _ = dst.shape
    inv = 1.0 / (s * s)
    for y in prange(h):
        for x in range(w):
            for c in range(3):
                a = 0.0
                for j in range(s):
                    for i in range(s):
                        a += src[y * s + j, x * s + i, c]
                dst[y, x, c] = a * inv


@njit(parallel=True, fastmath=True, cache=True)
def _blur(a, tmp, r):
    """Box blur, separable, radius r, in place (three passes make it near-Gaussian)."""
    h, w, _ = a.shape
    n = 1.0 / (2 * r + 1)
    for _ in range(3):
        for y in prange(h):
            for x in range(w):
                for c in range(3):
                    s = 0.0
                    for d in range(-r, r + 1):
                        s += a[y, min(max(x + d, 0), w - 1), c]
                    tmp[y, x, c] = s * n
        for y in prange(h):
            for x in range(w):
                for c in range(3):
                    s = 0.0
                    for d in range(-r, r + 1):
                        s += tmp[min(max(y + d, 0), h - 1), x, c]
                    a[y, x, c] = s * n


@njit(parallel=True, fastmath=True, cache=True)
def _compose(e, bloom, s, glow, exposure, lut, out):
    """Emission plus bloom (bilinear up), tone-mapped and gamma-encoded."""
    H, W, _ = e.shape
    h, w, _ = bloom.shape
    for y in prange(H):
        fy = (y + 0.5) / s - 0.5
        y0 = min(max(int(math.floor(fy)), 0), h - 1)
        y1 = min(y0 + 1, h - 1)
        ty = min(max(fy - y0, 0.0), 1.0)
        v = np.empty(3, np.float32)
        for x in range(W):
            fx = (x + 0.5) / s - 0.5
            x0 = min(max(int(math.floor(fx)), 0), w - 1)
            x1 = min(x0 + 1, w - 1)
            tx = min(max(fx - x0, 0.0), 1.0)
            for c in range(3):
                bl = (bloom[y0, x0, c] * (1 - tx) + bloom[y0, x1, c] * tx) * (1 - ty) \
                    + (bloom[y1, x0, c] * (1 - tx) + bloom[y1, x1, c] * tx) * ty
                v[c] = exposure * (e[y, x, c] + glow * bl)
            m = max(v[0], v[1], v[2])               # tone-map the brightest channel, keeping hue
            if m < 1e-6:
                out[y, x, 0] = out[y, x, 1] = out[y, x, 2] = 0
                continue
            t = (1.0 - math.exp(-m)) / m * (len(lut) - 1)
            for c in range(3):
                out[y, x, c] = lut[int(v[c] * t)]


class Camera:
    """Turns a grill into an RGB frame (H, W, 3) uint8."""

    def __init__(self, grill, s=4, radius=2, glow=1.0, exposure=0.15):
        H, W = grill.H, grill.W
        self.s, self.radius, self.glow, self.exposure = s, radius, glow, exposure
        self.e = np.zeros((H, W, 3), np.float32)
        self.small = np.zeros((H // s, W // s, 3), np.float32)
        self.tmp = np.zeros_like(self.small)
        self.rgb = np.zeros((H, W, 3), np.uint8)
        self.lut = np.round(255 * np.linspace(0, 1, 4096) ** (1 / 2.2)).astype(np.uint8)  # gamma

    def shoot(self, gr):
        _emit(gr.flame, gr.g, np.float32(gr.P['k'] / gr.P['q']), np.float32(gr.P['glean']), BLUE, SOOT, self.e)
        _down(self.e, self.small, self.s)
        _blur(self.small, self.tmp, self.radius)
        _compose(self.e, self.small, self.s, np.float32(self.glow), np.float32(self.exposure), self.lut, self.rgb)
        return self.rgb


# ---------------------------------------------------------------- front ends

def knob_at(schedule, t):
    """Piecewise-linear knob from 'v:t,v:t,...' (seconds)."""
    pts = [tuple(map(float, s.split(':'))) if ':' in s else (float(s), 0.0) for s in schedule.split(',')]
    if t <= pts[0][1]:
        return pts[0][0]
    for (v0, t0), (v1, t1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / max(t1 - t0, 1e-9)
    return pts[-1][0]


def make(a):
    W, H = map(int, a.size.split('x'))
    kw = dict(kv.split('=') for kv in a.set) if a.set else {}
    lattice = a.layout in ('grid', 'hex', 'random')
    gr = Grill(W, H, n=a.burners or (W * H // 324 if lattice else 10),
               fill=a.fill or (0.8 if lattice else 0.97), layout=a.layout, seed=a.seed,
               **{k: float(v) for k, v in kw.items()})
    gr.knob = knob_at(a.knob, 0)
    gr.strike(W // 2, H // 2)
    return gr


def measure(a):
    gr = make(a)
    on = gr.supply > 0
    t0 = time.time()
    for i in range(a.ticks // 100):
        gr.knob = knob_at(a.knob, i * 100 / (a.fps * a.speed))
        gr.run(100)
        if i % 10 == 9:
            print(f'tick {gr.t:6d}  knob {gr.knob:.2f}  lit {(gr.flame[on] > 0).mean():.3f}  '
                  f'gas {gr.g[on].mean():.2f}  heat(air) {gr.f[~on].mean():.3f}')
    dt = time.time() - t0
    print(f'{gr.W}x{gr.H}: {dt / gr.t * 1e3:.2f} ms/tick')
    if a.snap:
        import pygame
        rgb = Camera(gr).shoot(gr)
        pygame.image.save(pygame.image.frombuffer(rgb.tobytes(), (gr.W, gr.H), 'RGB'), a.snap)


class Video:
    """Frames piped to ffmpeg as they come, into an H.264 MP4."""

    def __init__(self, path, W, H, fps, preset='slow', threads=0):
        import imageio_ffmpeg
        self.path, self.fps, self.frames = path, fps, 0
        self.ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                    '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', preset,
                                    '-threads', str(threads), '-crf', '16', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)

    def write(self, rgb):
        self.ff.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def render(a):
    gr = make(a)
    cam = Camera(gr)
    video = Video(a.render, gr.W, gr.H, a.fps)
    n = int(a.seconds * a.fps)
    for i in range(n):
        gr.knob = knob_at(a.knob, i / a.fps)
        gr.run(a.speed)
        video.write(cam.shoot(gr))
        if i % a.fps == 0:
            print(f'{i / a.fps:5.1f}s  knob {gr.knob:.2f}', flush=True)
    video.close()


def watch(a):
    import pygame
    gr = make(a)
    cam = Camera(gr)
    pygame.init()
    scale = min(1.0, 1600 / gr.W, 900 / gr.H)
    vw, vh = int(gr.W * scale), int(gr.H * scale)
    bar = 40
    screen = pygame.display.set_mode((vw, vh + bar))
    pygame.display.set_caption('gas grill')
    font = pygame.font.SysFont('monospace', 16)
    kmax, speed, paused, drag = 2.0, a.speed, False, False
    clock = pygame.time.Clock()
    sx0, sx1 = 200, vw - 20
    video = None
    if not a.no_record:                                # every frame shown goes into a video
        os.makedirs('recordings', exist_ok=True)
        video = Video(time.strftime('recordings/grill-%Y%m%d-%H%M%S.mp4'), gr.W, gr.H, a.fps, 'veryfast', 2)  # light, to spare the simulation
    stop, frames, t0 = [], 0, time.time()
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))   # Ctrl-C quits too, keeping the video
    try:
        while not stop:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key in (pygame.K_UP, pygame.K_RIGHT):
                        gr.knob = min(kmax, gr.knob + 0.02)
                    elif e.key in (pygame.K_DOWN, pygame.K_LEFT):
                        gr.knob = max(0.0, gr.knob - 0.02)
                    elif e.key == pygame.K_EQUALS:
                        speed = speed * 2
                    elif e.key == pygame.K_MINUS:
                        speed = max(1, speed // 2)
                    elif e.key == pygame.K_r:
                        knob = gr.knob
                        a.seed += 1
                        gr = make(a)
                        gr.knob = knob
                if e.type == pygame.MOUSEBUTTONDOWN:
                    if e.pos[1] >= vh:
                        drag = True
                    else:
                        gr.strike(int(e.pos[0] / scale), int(e.pos[1] / scale))
                if e.type == pygame.MOUSEBUTTONUP:
                    drag = False
                if drag and e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                    gr.knob = kmax * min(max((e.pos[0] - sx0) / (sx1 - sx0), 0), 1)
            if not paused:
                gr.run(speed)
            rgb = cam.shoot(gr)
            frames += 1
            if frames == 20:                               # past start-up
                t0 = time.time()
            if video and not paused:
                video.write(rgb)
            img = pygame.image.frombuffer(rgb.tobytes(), (gr.W, gr.H), 'RGB')
            screen.blit(pygame.transform.smoothscale(img, (vw, vh)) if scale < 1 else img, (0, 0))
            pygame.draw.rect(screen, (20, 20, 20), (0, vh, vw, bar))
            rec = f'  REC {video.frames / video.fps:.0f}s' if video else ''
            screen.blit(font.render(f'knob {gr.knob:.2f}  x{speed}{rec}', True, (220, 220, 220)), (10, vh + 11))
            y = vh + bar // 2
            pygame.draw.line(screen, (120, 120, 120), (sx0, y), (sx1, y), 2)
            dx = sx0 + (sx1 - sx0) / kmax                      # the detent, at 1
            pygame.draw.line(screen, (200, 200, 200), (dx, y - 10), (dx, y + 10), 2)
            kx = sx0 + (sx1 - sx0) * gr.knob / kmax
            pygame.draw.circle(screen, (255, 255, 255), (int(kx), y), 8)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
        print(f'{(frames - 20) / (time.time() - t0):.0f} frames per second')
        if video:
            print(f'finishing {video.path} ({video.frames / video.fps:.1f}s of video) ...', flush=True)
            video.close()
            print(f'saved {video.path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render', metavar='OUT.mp4')
    ap.add_argument('--measure', action='store_true', help='headless: print how much is lit')
    ap.add_argument('--size', default='960x540')
    ap.add_argument('--layout', choices=('eight', 'honeycomb', 'grid', 'hex', 'random'), default='eight')
    ap.add_argument('--burners', type=int,
                    help='holes per loop of the 8 (default 10; honeycomb copies their size), or in all (default one per 18x18 cells)')
    ap.add_argument('--fill', type=float, help='hole diameter / spacing (default 0.97 for the 8, 0.8 otherwise)')
    ap.add_argument('--knob', default='1.5', help="value, or schedule 'v:t,v:t' in seconds")
    ap.add_argument('--speed', type=int, default=64, help='ticks per frame')
    ap.add_argument('--fps', type=int, default=30, help='video frames per second')
    ap.add_argument('--no-record', action='store_true', help='play without saving a video')
    ap.add_argument('--seconds', type=float, default=20)
    ap.add_argument('--ticks', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--snap', metavar='PNG', help='with --measure: save the last frame')
    ap.add_argument('--set', nargs='*', metavar='NAME=VALUE', help='override model parameters')
    a = ap.parse_args()
    if a.render:
        render(a)
    elif a.measure:
        measure(a)
    else:
        watch(a)


if __name__ == '__main__':
    main()
