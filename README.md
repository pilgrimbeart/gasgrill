# Gas grill

[![The figure-of-8 grill: fully lit, then the gas is turned down and flames start hopping from hole to hole](media/demo.gif)](media/demo.mp4)

*Click for the full-quality video ([media/demo.mp4](media/demo.mp4)).*

An eye-level gas grill, the kind used for toast and bacon, is a flat envelope fed with gas, with holes cut in its underside where the gas meets the air and burns. A detent on the control stops you turning the gas down below the point where every hole stays lit. Push past the detent and the grill starts organising itself:
- some holes go out;
- gas builds up at them;
- a neighbour's flame relights them, while the neighbour goes out in turn.

This project simulates that grill. It's the textbook example of an *excitable medium*: the same kind of system as heart muscle, nerve fibres and the Belousov–Zhabotinsky reaction. It is also the starting point of the [neuron2025](https://github.com/pilgrimbeart/neuron2025) project, which tries to grow a brain from a sheet of identical cells.

## How it works

The grill is a sheet of identical cells, 960×540 by default. Every cell runs the same small rule, at random moments; there's no global clock. The holes are discs of cells that are fed gas at a rate set by the knob. By default there are 19 large holes arranged in a figure of 8: 10 per loop, with one shared where the loops cross.

Each cell holds three numbers:
- **gas:** private to the cell;
- **flame size:** also private, from 0 (out) to 1 (full);
- **heat:** the only number the cell shares with its neighbours.

The rule:
1. **Lighting.** An unlit cell lights when its neighbours' heat times its own gas passes a threshold, as long as it holds more gas than the lean limit. A strong flame lights thin gas, but faint warmth only lights a rich pool.
2. **Burning.** A flame burns gas in proportion to its size and to the gas there.
   - It grows while it burns at least the *quench* rate.
   - It shrinks away when it burns less, so flames swell and die smoothly rather than switching on and off.
3. **Heat.** A burning cell gives off heat. Heat spreads to the neighbours and fades.

The detent isn't programmed in; it follows from this rule:
- **Above the detent** (knob above 1), a lit hole settles where it burns exactly what it's fed, which is above the quench rate, so every hole stays lit.
- **Below it**, a hole can't sustain its flame. It burns down its store, shrinks away and goes dark, then slowly refills. Once it holds enough gas, a burning neighbour relights it.
- Holes that happen to get a little more gas stay lit and act as pacemakers. Waves of flame travel around the loops and through the crossing.

The light is modelled on real flames:
- **Blue:** clean burning with plenty of air glows blue, from CH and C₂ molecules.
- **Yellow-orange:** a hole that has pooled gas for a long time lights rich and flashes yellow-orange, from glowing soot at about 1900 K, then burns down to blue.
- **Black:** unlit cells give no light.

## Play

```bash
.venv/bin/python grill.py
```

The grill starts at knob 1.5 with a match struck in the middle, so the whole figure of 8 lights up. Then:

1. **Turn the gas down** by dragging the slider at the bottom of the window to the left, past the tick mark (the detent, at 1.0).
2. **Watch** for emergent behaviour. Holes go out and refill, flames hop from hole to hole and run around the loops, and rich pools flare yellow. Just below the detent (about 0.9–0.97) is the most interesting range. Much lower and the grill simply goes out, as a real one does. Click on a hole to strike a match and light it again.
3. **Quit with Esc.** Everything you saw has been recorded. The terminal prints `saved recordings/grill-<date>-<time>.mp4`.

| Control | What it does |
|---|---|
| Drag the slider | Turn the gas (the tick mark is the detent) |
| ← / → (or ↓ / ↑) | Nudge the gas by 0.02 |
| Click on the grill | Strike a match |
| `=` / `-` | Double or halve the simulation speed |
| Space | Pause (not recorded) |
| `r` | Start again with a new random seed, keeping the knob |
| Esc, `q` or Ctrl-C | Quit and save the recording |

### Recordings

- **What's saved:** every frame shown is saved to `recordings/` as an H.264 MP4 at 960×540. Only the grill is included, not the knob bar.
- **Playback speed:** videos play at 30 frames per second, a little faster than most machines run live. `--fps 20` matches live speed more closely.
- **Not recording:** use `--no-record`.
- **On WSL:** `explorer.exe recordings/<file>.mp4` plays a recording in Windows' default player.

## Set up from scratch

You need Python 3.10 or later, and a desktop for the window. On Windows, WSL2 provides one through WSLg.

```bash
git clone https://github.com/pilgrimbeart/gasgrill.git
cd gasgrill
python3 -m venv .venv            # on Debian/Ubuntu, first: sudo apt install python3-venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python grill.py
```

The requirements are:
- `numpy`;
- `numba`, which compiles the cell rule;
- `pygame`, for the window;
- `imageio-ffmpeg`, which bundles an ffmpeg for the videos;
- `tbb`, which numba uses to start its threads quickly. `grill.py` loads it itself.

The first run takes a few seconds longer while numba compiles the rule. It keeps the compiled code for later runs.

## Other ways to run it

```bash
# A video made offline, with the gas on a schedule (value:seconds, linear between)
.venv/bin/python grill.py --render out.mp4 --seconds 40 --knob 1.5:0,1.5:10,0.95:14

# Full HD (slower)
.venv/bin/python grill.py --size 1920x1080

# The 8's holes, at the same size and spacing, packed hexagonally over the whole sheet
.venv/bin/python grill.py --layout honeycomb

# A sheet of thousands of small holes instead of the figure of 8 (also grid, random)
.venv/bin/python grill.py --layout hex

# More holes per loop, or wider gaps between them
.venv/bin/python grill.py --burners 14 --fill 0.9

# Headless: print how much is lit as it runs
.venv/bin/python grill.py --measure --knob 1.5:0,1.5:1,0.95:2 --ticks 10000
```

`--set NAME=VALUE` overrides any of the model's parameters, which are listed with explanations at the top of `grill.py`. `--help` lists every option.
