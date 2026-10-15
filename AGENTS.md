# Agent Notes

This is a private project. No one else works on it. Always work directly on `main`. Commit and push only when the user says "Push".

Prefer a complete rewrite with breaking changes when it gives a cleaner, smaller design. Minimalism and a single obvious architecture beat compatibility.

## What this is

A simulator of an eye-level gas grill run below its detent, to make high-resolution video for a talk about excitable media. The talk's story runs from the grill to the user's neuron2025 project (`../neuron2025`), which this is the ancestor of.

## The bigger picture

The talk follows the user's exploration of excitable media, from a toy to an attempt at a brain:

1. **The gas grill.** Simple local burning, coupled by neighbours lighting neighbours, organises itself into waves and pacemakers. This repo.
2. **A designed excitable medium.** In `../neuron2025/old_manual_gates/`, an excitable medium was designed by hand, with gates built into it to compute. It still runs (`cd old_manual_gates && python neuron.py`).
3. **neuron2025** (`../neuron2025`, see its README and LESSONS.md). The aim is *emergent learning*: a sheet of identical cells, each running the same small local rule, that builds itself into a brain and learns within its lifetime. The test is a creature learning which colour is food and which is poison.
   - An evolved cell rule learned to move and graze, but never which colour is food.
   - Hand-designed development in stages does learn it. Cells lay down a coordinate frame, grow wiring, move, and learn.
4. **nca-learn** (`../nca-learn`). The same creature, under the strict rule that each cell sees only the average of its neighbours' shared numbers.

This grill sim should therefore look like the first step on that road. It uses identical cells, one local rule, and as few shared numbers as possible (here one), with noise and random update times. It is not a fluid-dynamics simulation. The interesting behaviour must emerge from the cells, not be scripted by the harness.

Treat `../neuron2025` and `../nca-learn` as read-only reference. The user will supply the talk's own structure. Ask rather than guess.

## The grill

The real grill is a flat envelope fed with gas, with holes cut in its underside where the gas meets the air and burns. A detent stops the flow being turned below the level at which every hole stays lit. Below it the grill organises itself. Some holes go out, gas builds up at them, and a neighbour's flame relights them, while the neighbour goes out in turn.

## The model (agreed design)

A sheet of identical cells, each running one small local rule.

- **Burners:** circular discs (the holes), with count, size and layout as parameters. The default layout is a horizontal figure of 8, with 10 large holes per loop and one shared where the loops cross. `honeycomb` packs the same holes, at the same spacing, hexagonally over the whole sheet. A sheet of many small holes (hex, grid or random) is also available. Gas enters uniformly over each disc at the supply rate `s`, which is **the knob**. The space between discs is air.
- **Each cell holds:**
  - gas `g` and a flame size (0 when out, up to 1), both private;
  - heat `f`, the **one shared number** ("ignition"), which spreads to neighbours and decays.
- **The rule:**
  1. An unlit cell holding more than the lean limit lights when its neighbours' heat times its gas passes its threshold. A strong flame lights thin gas; faint warmth lights only a rich pool. A new flame starts small.
  2. A flame burns size × `k` × `g`. It grows while that is at least `q` and shrinks away when it falls below, so flames swell and die over many updates, never in one frame.
- **The detent falls out of this:**
  - At `s > q` a lit burner settles where burn rate equals supply, and everything stays lit.
  - At `s < q` it burns its store down, goes out, and refills past the lean limit (its refractory time) until a neighbour relights it.
  - The burners that reach flammability first become pacemakers, driving target waves and spirals.
- This is the Barkley / FitzHugh–Nagumo shape: only the fast variable diffuses, and the recovery variable (gas) stays local.
- **Start with the one-shared-number version.** A second shared number would let unburnt gas spread into the gaps as a cloud. The real grill does this: an unlit hole leaks gas that pools until a neighbour's flame reaches it with a "whoomph". Add it only if the video needs it.

Carried over from neuron2025:
- Cells update at random times (each tick, each cell updates with some probability), never in lockstep.
- Noise is real: there is fixed per-cell variation in the thresholds, which picks the pacemakers, plus small noise over time. Rules must be robust to it.
- The harness is only the world: the gas supply and the knob.

## Colour: as realistic as possible

Model the light from the physics, not a hue mapped onto gas level:

- **Blue:** premixed, well-aired burning glows blue from CH (~431 nm) and C2 (~516 nm) emission. A steady flame with enough air is blue, a lean or dying one is dim blue, and a very lean one is nearly invisible.
- **Yellow-orange:** fuel-rich, poorly mixed burning glows from incandescent soot at about 1800–2000 K. A pool of accumulated gas flaring up should therefore flash **yellow-orange**, and then turn blue as it burns down to steady or lean.
- **Light model:** blue emission scales with burn rate, and soot emission scales with burn rate times how fuel-rich the cell is (`g` relative to the steady level). The two add as light on a dark background (the grill's underside). A little bloom or glow is fine for video.
- **Unlit cells are pure black**, for maximum contrast (the user's choice; this replaced an earlier faint haze showing unburnt gas).
- **Greyscale check:** the user is colour-blind. The rendering must read in greyscale: brightness has to carry what matters, not hue alone. Blue against yellow is the safest hue pair, and yellow is naturally brighter.

## Performance and output

- **Resolution:** 16:9 to match slides. The default is 960×540 cells (1920×1080 chugged live); `--size` changes it. It must be very efficient, so use numpy, with numba kernels for the cell update.
- **Live view:** the default mode (`python grill.py`), a window with a knob slider for exploring the settings.
- **Video:** playing records every frame shown to `recordings/grill-<time>.mp4`, saved on Esc (`--no-record` to skip), so the knob can be worked by hand. `--render` makes a deterministic, seeded MP4 offline, with the knob on a schedule.
- **Later, maybe:** a WebGL browser version, so the knob can be turned live on stage. Not now.
- Before reporting something as working, show it live in the interactive view.
