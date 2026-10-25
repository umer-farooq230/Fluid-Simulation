# Fluid Simulation in Python

A real-time 2D fluid you can grab, lift, and splash with your mouse. About 9,000 particles, smooth on a laptop, with a side menu that changes the fluid while it runs.

https://github.com/umer-farooq230/Fluid-Simulation/blob/main/PBF%20Fluid%202026-10-18%2002-07-04.mp4

> It started at **~10 FPS**. Making it fast taught me more about optimization than any course did. I wrote about it here: **[Understanding the Math Behind Fluid Simulations by Coding in Python](https://medium.com/towardsdev/understanding-the-math-behind-fluid-simulation-by-coding-in-python-a21c530dfcf0)**

---

## Run it

```bash
pip install numpy pygame numba
python fluid_app.py              # full screen
python fluid_app.py --windowed   # normal window
```

Press **Start** in the side menu (or `P`). The first launch takes a few seconds while Numba compiles the physics; after that it's cached.

## Controls

| Action | What it does |
|---|---|
| **Double-click** | Latch onto the fluid inside the circle. Move to lift it, **click once** to drop it and watch it splash |
| **Hold left mouse** | Grab only while held |
| **Hold `Space`** (or right mouse) | Push the fluid away |
| **Mouse wheel** / `[` `]` | Change the hand size |
| `P` start/pause, `G` gravity, `R` restart, `H` info box, `Esc` quit | |

Only particles **inside the circle** get picked up.

The side menu has presets (Water, Syrup, Honey, Moon) and sliders with plain words like "syrupy" or "firm", so you can feel what each setting does without knowing the math.

## How it works (the intuition)

Every particle asks one question each step: **"am I more crowded than I should be?"**

1. **Predict** where each particle will be next (current velocity + gravity).
2. **Measure crowding** (density) by looking at nearby neighbors. Closer neighbors count more (the smoothing kernel).
3. **Push apart** the ones that are too crowded, a little, over a few passes. This is the same loop as gradient descent: measure the error, step downhill, repeat.
4. **Velocity = how far it actually moved.** That's what makes the fluid flow and splash.

## Why it's fast

| Problem | Fix |
|---|---|
| Every particle checked every other: ~80 million comparisons per step | **Spatial grid.** Cells as wide as the smoothing radius, so neighbors can only be in the 3×3 cells around you. Cost grows with N, not N² |
| Python loops are slow | **Numba** compiles the hot loops to machine code and runs them on all cores |
| Small time steps to stay stable | The position-based solver stays stable with **large steps** |
| Drawing 9,000 circles | Write pixels **directly** into the screen buffer |

## Files

```
fluid_backend.py   the physics: kernels, grid, solver, settings and presets
fluid_app.py       the window, side menu and mouse controls
```

The backend has no UI code, so you can import `FluidSimulation` and use it in your own project.

## Ideas to try

- Turn gravity off and use the hand to stir.
- Compare Water and Honey: the only difference is the thickness slider.
- Raise firmness for a fluid that resists squishing, lower it for a softer one.

## Credits

This project was inspired by [Sebastian Lague: Coding Adventures: Simulating Fluids](https://www.youtube.com/watch?v=rSKMYc1CQHE)
