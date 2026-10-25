"""
Fast interactive 2D fluid (Position Based Fluids) with gravity and mouse grab.

Install: pip install numpy pygame numba
Run:     python fluid_grid.py

Controls (laptop friendly - no right/middle button needed)
  DOUBLE-CLICK      latch onto the fluid inside the circle; move it (lift it up),
                    then click once to let go and watch it splash
  Hold LEFT mouse   (or tap-and-drag) grabs only while held
  Hold SPACE        push fluid away from the cursor (RIGHT mouse works too)
  Mouse wheel       change grab / push radius  (or [ and ] keys)
  G                 toggle gravity
  R                 reset      H  toggle help      ESC  quit

Only the particles inside the circle at the moment you press are held; they are
drawn WHITE so you can see exactly which ones. Everything else is ordinary
fluid that simply reacts to them.

Where the fluid maths lives (see the notes at the bottom of this file too):
  gravity / mouse forces     integrate
  density + gradient         solve_lambda
  pressure push-apart        compute_dp / apply_dp
  incompressibility          the SOLVER_ITERS loop inside step()
  viscosity                  xsph
"""
import math

import numpy as np
import pygame
from numba import njit, prange

# ----------------------------------------------------------------- settings
SPACING = 11.0            # particle spacing in px (smaller = more particles)
SMOOTHING_RADIUS = 2.2 * SPACING   # how far a particle "sees" its neighbours
MASS = SPACING * SPACING  # makes rest density ~ 1
GRAVITY = 2200.0          # px / s^2
SOLVER_ITERS = 4          # constraint iterations per substep
SUB_DT = 1.0 / 90.0       # target substep
MAX_SUBSTEPS = 2
MAX_SPEED = 1600.0
VISCOSITY = 0.08          # XSPH coefficient
FILL_FRACTION = 0.55      # initial pool height as fraction of screen
WALL = 8.0

GRAB_RADIUS = 110.0
GRAB_GAIN = 12.0          # 1/s, how hard grabbed particles chase the cursor
GRAB_BLEND = 25.0         # 1/s, how fast velocity is steered
MAX_GRAB_SPEED = 2200.0
PUSH_STRENGTH = 9000.0
DOUBLE_CLICK_MS = 400     # two clicks within this time = double-click
DOUBLE_CLICK_PX = 30      # ...and within this many pixels

BG = (8, 11, 18)
COLOR_SCALE = 1.0 / 900.0   # speed of 900 px/s = full red


# ------------------------------------------- SPH building blocks (reference)
@njit(cache=True)
def smoothing_kernel(radius, distance):
    """2D poly6 kernel. Works for a scalar distance (inside the compiled
    solver) and for a NumPy array of distances (from normal Python)."""
    volume = math.pi * radius ** 8 / 4.0
    value = np.maximum(0.0, radius * radius - distance * distance)
    return value ** 3 / volume


def calculate_density(positions, sample_point, smoothing_radius):
    """Density at sample_point = sum_j m * W(|x_j - sample_point|)."""
    distances = np.linalg.norm(positions - sample_point, axis=1)
    return float(np.sum(MASS * smoothing_kernel(smoothing_radius, distances)))


def calculate_property(particle_properties, sample_point, positions,
                       smoothing_radius, densities):
    """SPH interpolation: A(x) = sum_j A_j * (m / rho_j) * W(|x_j - x|).
    `densities` is computed once per particle by the caller (never inside
    this loop - that made the old version O(N^2) per call)."""
    distances = np.linalg.norm(positions - sample_point, axis=1)
    kernel_weights = smoothing_kernel(smoothing_radius, distances)
    weights = kernel_weights * MASS / np.maximum(densities, 1e-8)
    return float(np.sum(particle_properties * weights))


def CreateParticles(seed, origin, boundsize, spacing=SPACING):
    """Evenly spaced (slightly jittered) particles inside a rectangle.
    Uniform random placement gives clumps and gaps, i.e. wildly wrong
    starting densities and an explosion on frame one, so we use a grid."""
    random_generator = np.random.default_rng(seed)
    xs = np.arange(0.0, boundsize[0], spacing)
    ys = np.arange(0.0, boundsize[1], spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions = np.column_stack((grid_x.ravel(), grid_y.ravel())) + np.asarray(origin)
    positions += random_generator.uniform(-0.03, 0.03, positions.shape) * spacing
    particle_properties = np.zeros(len(positions))
    return positions.astype(np.float64), particle_properties


# ------------------------------------------------------------ numba kernels
@njit(cache=True)
def build_grid(pred, inv_cell_size, grid_cols, grid_rows, cell_of, cell_start, order):
    """Sort particles into square cells (cell size = smoothing radius) so a
    particle only has to look at the 3x3 cells around it for neighbours.
    Particles of cell c are order[cell_start[c] : cell_start[c + 1]]."""
    count = pred.shape[0]
    cell_start[:] = 0
    for particle in range(count):
        cell_x = min(max(int(pred[particle, 0] * inv_cell_size), 0), grid_cols - 1)
        cell_y = min(max(int(pred[particle, 1] * inv_cell_size), 0), grid_rows - 1)
        cell = cell_y * grid_cols + cell_x
        cell_of[particle] = cell
        cell_start[cell + 1] += 1
    for cell in range(grid_cols * grid_rows):
        cell_start[cell + 1] += cell_start[cell]
    next_slot = cell_start[:-1].copy()
    for particle in range(count):
        cell = cell_of[particle]
        order[next_slot[cell]] = particle
        next_slot[cell] += 1


@njit(parallel=True, cache=True, fastmath=True)
def integrate(pos, vel, pred, dt, gravity, grabbed, grab_offset, mouse_x, mouse_y,
              grab_on, grab_gain, grab_blend, max_grab,
              push_on, push_radius, push_strength, max_speed):
    """Apply gravity + mouse forces to the velocities, then predict where each
    particle will be after dt (pred = pos + vel * dt)."""
    count = pos.shape[0]
    blend = min(1.0, grab_blend * dt)
    for particle in prange(count):
        vel_x = vel[particle, 0]
        vel_y = vel[particle, 1] + gravity * dt           # gravity term
        pos_x = pos[particle, 0]
        pos_y = pos[particle, 1]

        # grab: steer only the held particles towards "cursor + their offset"
        if grab_on and grabbed[particle]:
            target_vel_x = (mouse_x + grab_offset[particle, 0] - pos_x) * grab_gain
            target_vel_y = (mouse_y + grab_offset[particle, 1] - pos_y) * grab_gain
            target_speed = math.sqrt(target_vel_x * target_vel_x + target_vel_y * target_vel_y)
            if target_speed > max_grab:
                target_vel_x *= max_grab / target_speed
                target_vel_y *= max_grab / target_speed
            vel_x += (target_vel_x - vel_x) * blend
            vel_y += (target_vel_y - vel_y) * blend

        # push: accelerate away from the cursor, strongest at the centre
        if push_on:
            away_x = pos_x - mouse_x
            away_y = pos_y - mouse_y
            distance = math.sqrt(away_x * away_x + away_y * away_y)
            if 1e-6 < distance < push_radius:
                falloff = 1.0 - distance / push_radius
                push = falloff * falloff * push_strength * dt
                vel_x += away_x / distance * push
                vel_y += away_y / distance * push

        speed = math.sqrt(vel_x * vel_x + vel_y * vel_y)
        if speed > max_speed:
            vel_x *= max_speed / speed
            vel_y *= max_speed / speed

        vel[particle, 0] = vel_x
        vel[particle, 1] = vel_y
        pred[particle, 0] = pos_x + vel_x * dt
        pred[particle, 1] = pos_y + vel_y * dt


@njit(parallel=True, cache=True, fastmath=True)
def solve_lambda(pred, order, cell_start, grid_cols, grid_rows, inv_cell_size,
                 smoothing_radius, mass, inv_rest_density, softening,
                 lagrange, densities):
    """Measure each particle's density and compute its Lagrange multiplier.

    densities[p]  = density at the predicted position of particle p
    lagrange[p]   = 0 if not crowded, otherwise a negative number: how hard
                    particle p must be pushed so its density returns to rest.
    softening     tiny number added to the denominator for stability.
    Only compression (density above rest) is resisted."""
    count = pred.shape[0]
    radius_squared = smoothing_radius * smoothing_radius
    spiky = -30.0 / (math.pi * smoothing_radius ** 5)   # gradient kernel constant
    for particle in prange(count):
        particle_x = pred[particle, 0]
        particle_y = pred[particle, 1]
        cell_x = min(max(int(particle_x * inv_cell_size), 0), grid_cols - 1)
        cell_y = min(max(int(particle_y * inv_cell_size), 0), grid_rows - 1)
        # density is evaluated at the PREDICTED positions (pred), not pos
        density = mass * smoothing_kernel(smoothing_radius, 0.0)
        grad_sum_x = 0.0           # sum of kernel gradients over all neighbours
        grad_sum_y = 0.0
        neighbor_grad_sq_sum = 0.0
        for cell_row in range(max(cell_y - 1, 0), min(cell_y + 2, grid_rows)):
            for cell_col in range(max(cell_x - 1, 0), min(cell_x + 2, grid_cols)):
                cell = cell_row * grid_cols + cell_col
                for slot in range(cell_start[cell], cell_start[cell + 1]):
                    neighbor = order[slot]
                    if neighbor == particle:
                        continue
                    offset_x = particle_x - pred[neighbor, 0]
                    offset_y = particle_y - pred[neighbor, 1]
                    distance_squared = offset_x * offset_x + offset_y * offset_y
                    if distance_squared < radius_squared:
                        distance = math.sqrt(distance_squared)
                        density += mass * smoothing_kernel(smoothing_radius, distance)
                        if distance > 1e-9:
                            overlap = smoothing_radius - distance
                            grad_scale = spiky * overlap * overlap * mass * inv_rest_density / distance
                            grad_x = grad_scale * offset_x       # gradient of the density
                            grad_y = grad_scale * offset_y       # constraint w.r.t. neighbour
                            grad_sum_x += grad_x
                            grad_sum_y += grad_y
                            neighbor_grad_sq_sum += grad_x * grad_x + grad_y * grad_y
        densities[particle] = density
        density_error = density * inv_rest_density - 1.0      # > 0 means crowded
        if density_error > 0.0:
            lagrange[particle] = -density_error / (
                neighbor_grad_sq_sum + grad_sum_x * grad_sum_x
                + grad_sum_y * grad_sum_y + softening)
        else:
            lagrange[particle] = 0.0


@njit(parallel=True, cache=True, fastmath=True)
def compute_dp(pred, order, cell_start, grid_cols, grid_rows, inv_cell_size,
               smoothing_radius, mass, inv_rest_density, lagrange, correction,
               max_correction):
    """Turn the multipliers into a "move by (x, y)" for every particle: the
    pressure push-apart. correction[p] = sum over neighbours of
    (lagrange[p] + lagrange[neighbour]) * kernel gradient."""
    count = pred.shape[0]
    radius_squared = smoothing_radius * smoothing_radius
    spiky = -30.0 / (math.pi * smoothing_radius ** 5)
    for particle in prange(count):
        particle_x = pred[particle, 0]
        particle_y = pred[particle, 1]
        own_lagrange = lagrange[particle]
        cell_x = min(max(int(particle_x * inv_cell_size), 0), grid_cols - 1)
        cell_y = min(max(int(particle_y * inv_cell_size), 0), grid_rows - 1)
        move_x = 0.0
        move_y = 0.0
        for cell_row in range(max(cell_y - 1, 0), min(cell_y + 2, grid_rows)):
            for cell_col in range(max(cell_x - 1, 0), min(cell_x + 2, grid_cols)):
                cell = cell_row * grid_cols + cell_col
                for slot in range(cell_start[cell], cell_start[cell + 1]):
                    neighbor = order[slot]
                    if neighbor == particle:
                        continue
                    offset_x = particle_x - pred[neighbor, 0]
                    offset_y = particle_y - pred[neighbor, 1]
                    distance_squared = offset_x * offset_x + offset_y * offset_y
                    if distance_squared < radius_squared and distance_squared > 1e-18:
                        distance = math.sqrt(distance_squared)
                        overlap = smoothing_radius - distance
                        grad_scale = spiky * overlap * overlap * mass * inv_rest_density / distance
                        pair_lagrange = own_lagrange + lagrange[neighbor]
                        move_x += pair_lagrange * grad_scale * offset_x
                        move_y += pair_lagrange * grad_scale * offset_y
        move_length = math.sqrt(move_x * move_x + move_y * move_y)
        if move_length > max_correction:
            move_x *= max_correction / move_length
            move_y *= max_correction / move_length
        correction[particle, 0] = move_x
        correction[particle, 1] = move_y


@njit(parallel=True, cache=True)
def apply_dp(pred, correction, x_min, x_max, y_min, y_max):
    """Move the predicted positions by the corrections (staying inside walls)."""
    for particle in prange(pred.shape[0]):
        new_x = pred[particle, 0] + correction[particle, 0]
        new_y = pred[particle, 1] + correction[particle, 1]
        pred[particle, 0] = min(max(new_x, x_min), x_max)
        pred[particle, 1] = min(max(new_y, y_min), y_max)


@njit(parallel=True, cache=True)
def clamp_pred(pred, x_min, x_max, y_min, y_max):
    for particle in prange(pred.shape[0]):
        pred[particle, 0] = min(max(pred[particle, 0], x_min), x_max)
        pred[particle, 1] = min(max(pred[particle, 1], y_min), y_max)


@njit(parallel=True, cache=True)
def velocity_from_positions(pos, pred, vel, inv_dt):
    """velocity = how far the particle actually moved / elapsed time. The
    pressure push-aparts become real motion here."""
    for particle in prange(pos.shape[0]):
        vel[particle, 0] = (pred[particle, 0] - pos[particle, 0]) * inv_dt
        vel[particle, 1] = (pred[particle, 1] - pos[particle, 1]) * inv_dt


@njit(parallel=True, cache=True, fastmath=True)
def xsph(pred, vel, smoothed_vel, order, cell_start, grid_cols, grid_rows,
         inv_cell_size, smoothing_radius, mass, inv_rest_density, viscosity_coeff):
    """XSPH viscosity: nudge each velocity towards its neighbours' average."""
    count = pred.shape[0]
    radius_squared = smoothing_radius * smoothing_radius
    for particle in prange(count):
        particle_x = pred[particle, 0]
        particle_y = pred[particle, 1]
        cell_x = min(max(int(particle_x * inv_cell_size), 0), grid_cols - 1)
        cell_y = min(max(int(particle_y * inv_cell_size), 0), grid_rows - 1)
        vel_diff_x = 0.0
        vel_diff_y = 0.0
        for cell_row in range(max(cell_y - 1, 0), min(cell_y + 2, grid_rows)):
            for cell_col in range(max(cell_x - 1, 0), min(cell_x + 2, grid_cols)):
                cell = cell_row * grid_cols + cell_col
                for slot in range(cell_start[cell], cell_start[cell + 1]):
                    neighbor = order[slot]
                    if neighbor == particle:
                        continue
                    offset_x = particle_x - pred[neighbor, 0]
                    offset_y = particle_y - pred[neighbor, 1]
                    distance_squared = offset_x * offset_x + offset_y * offset_y
                    if distance_squared < radius_squared:
                        distance = math.sqrt(distance_squared)
                        weight = mass * inv_rest_density * smoothing_kernel(smoothing_radius, distance)
                        vel_diff_x += (vel[neighbor, 0] - vel[particle, 0]) * weight
                        vel_diff_y += (vel[neighbor, 1] - vel[particle, 1]) * weight
        smoothed_vel[particle, 0] = vel[particle, 0] + viscosity_coeff * vel_diff_x
        smoothed_vel[particle, 1] = vel[particle, 1] + viscosity_coeff * vel_diff_y


@njit(parallel=True, cache=True)
def render(pixels, pos, vel, inv_color_scale, screen_width, screen_height,
           grabbed, highlight_grabbed):
    """Paint each particle as a 3x3 square straight into the pixel array.
    Colour by speed: slow blue -> cyan -> yellow -> fast red. Held particles
    are drawn white when highlight_grabbed is True."""
    count = pos.shape[0]
    for particle in prange(count):
        speed = math.sqrt(vel[particle, 0] ** 2 + vel[particle, 1] ** 2)
        speed_fraction = min(max(speed * inv_color_scale, 0.0), 1.0)   # 0 slow .. 1 fast
        if speed_fraction < 1.0 / 3.0:                    # blue -> cyan
            blend = speed_fraction * 3.0
            red = 35.0 - 35.0 * blend
            green = 105.0 + 120.0 * blend
            blue = 255.0 - 20.0 * blend
        elif speed_fraction < 2.0 / 3.0:                  # cyan -> yellow
            blend = (speed_fraction - 1.0 / 3.0) * 3.0
            red = 255.0 * blend
            green = 225.0 - 5.0 * blend
            blue = 235.0 - 195.0 * blend
        else:                                             # yellow -> red
            blend = (speed_fraction - 2.0 / 3.0) * 3.0
            red = 255.0
            green = 220.0 - 185.0 * blend
            blue = 40.0 + 5.0 * blend

        if highlight_grabbed and grabbed[particle]:
            red = 255.0
            green = 255.0
            blue = 255.0

        center_x = int(pos[particle, 0])
        center_y = int(pos[particle, 1])
        for offset_x in range(-1, 2):
            pixel_x = center_x + offset_x
            if pixel_x < 0 or pixel_x >= screen_width:
                continue
            for offset_y in range(-1, 2):
                pixel_y = center_y + offset_y
                if pixel_y < 0 or pixel_y >= screen_height:
                    continue
                pixels[pixel_x, pixel_y, 0] = np.uint8(red)
                pixels[pixel_x, pixel_y, 1] = np.uint8(green)
                pixels[pixel_x, pixel_y, 2] = np.uint8(blue)


# --------------------------------------------------------------- helpers
def lattice_stats(spacing, smoothing_radius, mass):
    """Rest density and typical constraint-gradient size on a square lattice."""
    reach = int(math.ceil(smoothing_radius / spacing))
    spiky = -30.0 / (math.pi * smoothing_radius ** 5)
    rest_density = 0.0
    for step_x in range(-reach, reach + 1):
        for step_y in range(-reach, reach + 1):
            distance_squared = (step_x * spacing) ** 2 + (step_y * spacing) ** 2
            if distance_squared < smoothing_radius ** 2:
                rest_density += mass * float(
                    smoothing_kernel(smoothing_radius, math.sqrt(distance_squared)))
    gradient_sq_sum = 0.0
    for step_x in range(-reach, reach + 1):
        for step_y in range(-reach, reach + 1):
            distance = math.hypot(step_x * spacing, step_y * spacing)
            if 0 < distance < smoothing_radius:
                gradient_sq_sum += (spiky * (smoothing_radius - distance) ** 2
                                    * mass / rest_density) ** 2
    return rest_density, gradient_sq_sum


def spawn(width, height):
    y_top = height * (1.0 - FILL_FRACTION)
    origin = (WALL + 2, y_top)
    boundsize = (width - 2 * (WALL + 2), height - WALL - 2 - y_top)
    positions, _ = CreateParticles(None, origin, boundsize)
    return positions


def main():
    pygame.init()
    pygame.display.set_caption("PBF Fluid")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    screen_w, screen_h = screen.get_size()

    # first run compiles the kernels (cached afterwards)
    screen.fill(BG)
    screen.blit(font.render("Compiling kernels (first run only)...", True,
                            (200, 215, 240)), (30, 30))
    pygame.display.flip()

    pos = spawn(screen_w, screen_h)
    particle_count = len(pos)
    vel = np.zeros_like(pos)
    pred = pos.copy()
    correction = np.zeros_like(pos)
    smoothed_vel = np.zeros_like(pos)
    lagrange = np.zeros(particle_count)
    densities = np.zeros(particle_count)
    grabbed = np.zeros(particle_count, dtype=np.bool_)
    grab_offset = np.zeros_like(pos)

    inv_cell_size = 1.0 / SMOOTHING_RADIUS
    grid_cols = int(screen_w * inv_cell_size) + 2
    grid_rows = int(screen_h * inv_cell_size) + 2
    cell_of = np.zeros(particle_count, dtype=np.int64)
    cell_start = np.zeros(grid_cols * grid_rows + 1, dtype=np.int64)
    order = np.zeros(particle_count, dtype=np.int64)

    rest_density, rest_gradient_sq_sum = lattice_stats(SPACING, SMOOTHING_RADIUS, MASS)
    inv_rest_density = 1.0 / rest_density
    softening = 0.3 * rest_gradient_sq_sum
    max_correction = 0.3 * SMOOTHING_RADIUS
    x_min, x_max = WALL, screen_w - WALL
    y_min, y_max = WALL, screen_h - WALL

    state = {"gravity": True, "help": True, "radius": GRAB_RADIUS,
             "grab": False, "latched": False,
             "last_click_t": -10**9, "last_click_xy": (0, 0),
             "ignore_until": 0}

    def start_grab(mouse_x, mouse_y):
        """Hold ONLY the particles currently inside the circle."""
        offset_from_mouse = pos - np.array([mouse_x, mouse_y])
        distance_squared = (offset_from_mouse[:, 0] ** 2
                            + offset_from_mouse[:, 1] ** 2)
        grabbed[:] = distance_squared < state["radius"] ** 2
        grab_offset[:] = offset_from_mouse
        state["grab"] = bool(grabbed.any())

    def release():
        state["grab"] = False
        state["latched"] = False
        grabbed[:] = False

    def step(dt, mouse_x, mouse_y, push_on):
        gravity = GRAVITY if state["gravity"] else 0.0
        integrate(pos, vel, pred, dt, gravity, grabbed, grab_offset, mouse_x, mouse_y,
                  state["grab"], GRAB_GAIN, GRAB_BLEND, MAX_GRAB_SPEED,
                  push_on, state["radius"], PUSH_STRENGTH, MAX_SPEED)
        clamp_pred(pred, x_min, x_max, y_min, y_max)
        build_grid(pred, inv_cell_size, grid_cols, grid_rows, cell_of, cell_start, order)
        for _ in range(SOLVER_ITERS):
            solve_lambda(pred, order, cell_start, grid_cols, grid_rows, inv_cell_size,
                         SMOOTHING_RADIUS, MASS, inv_rest_density, softening,
                         lagrange, densities)
            compute_dp(pred, order, cell_start, grid_cols, grid_rows, inv_cell_size,
                       SMOOTHING_RADIUS, MASS, inv_rest_density, lagrange,
                       correction, max_correction)
            apply_dp(pred, correction, x_min, x_max, y_min, y_max)
        velocity_from_positions(pos, pred, vel, 1.0 / dt)
        xsph(pred, vel, smoothed_vel, order, cell_start, grid_cols, grid_rows,
             inv_cell_size, SMOOTHING_RADIUS, MASS, inv_rest_density, VISCOSITY)
        vel[:] = smoothed_vel
        pos[:] = pred

    # warm-up (triggers compilation)
    step(SUB_DT, 0.0, 0.0, False)
    warmup_pixels = pygame.surfarray.pixels3d(screen)
    render(warmup_pixels, pos, vel, COLOR_SCALE, screen_w, screen_h, grabbed, False)
    del warmup_pixels
    pos[:] = spawn(screen_w, screen_h)
    vel[:] = 0

    running = True
    while running:
        frame_dt = min(clock.tick(120) / 1000.0, 1.0 / 30.0)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    state["help"] = not state["help"]
                elif event.key == pygame.K_g:
                    state["gravity"] = not state["gravity"]
                elif event.key == pygame.K_r:
                    pos[:] = spawn(screen_w, screen_h)
                    vel[:] = 0
                    release()
                elif event.key == pygame.K_LEFTBRACKET:
                    state["radius"] = max(40.0, state["radius"] - 10.0)
                elif event.key == pygame.K_RIGHTBRACKET:
                    state["radius"] = min(400.0, state["radius"] + 10.0)
            elif event.type == pygame.MOUSEWHEEL:
                state["radius"] = float(np.clip(
                    state["radius"] + event.y * 10.0, 40.0, 400.0))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # event.pos is exactly where the click happened
                click_x, click_y = event.pos
                now = pygame.time.get_ticks()
                if now < state["ignore_until"]:
                    pass
                elif state["latched"]:
                    # a click while latched lets go -> the held fluid keeps its
                    # velocity and splashes down under gravity
                    release()
                    state["ignore_until"] = now + DOUBLE_CLICK_MS
                else:
                    last_x, last_y = state["last_click_xy"]
                    is_double = (now - state["last_click_t"] <= DOUBLE_CLICK_MS
                                 and abs(click_x - last_x) <= DOUBLE_CLICK_PX
                                 and abs(click_y - last_y) <= DOUBLE_CLICK_PX)
                    start_grab(click_x, click_y)
                    state["latched"] = is_double and state["grab"]
                    state["last_click_t"] = -10**9 if is_double else now
                    state["last_click_xy"] = (click_x, click_y)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if not state["latched"]:
                    release()

        # the grab ends on mouse-UP (handled above), not by polling the button:
        # the polled state can lag one frame behind the click event.
        # Safety net: if the window lost focus we may never see the mouse-up.
        if not pygame.mouse.get_focused() and not state["latched"]:
            release()

        pressed_keys = pygame.key.get_pressed()
        push_on = bool(mouse_buttons[2] or pressed_keys[pygame.K_SPACE])

        substep_count = int(min(MAX_SUBSTEPS, max(1, round(frame_dt / SUB_DT))))
        dt = frame_dt / substep_count
        for _ in range(substep_count):
            step(dt, float(mouse_x), float(mouse_y), push_on)

        screen.fill(BG)
        pixels = pygame.surfarray.pixels3d(screen)
        render(pixels, pos, vel, COLOR_SCALE, screen_w, screen_h, grabbed, state["grab"])
        del pixels

        if state["grab"] or push_on:
            ring_color = (255, 220, 40) if state["latched"] else (120, 160, 220)
            pygame.draw.circle(screen, ring_color, (mouse_x, mouse_y),
                               int(state["radius"]), 1)

        if state["help"]:
            # cursor probe using the reference SPH functions above
            nearby = np.flatnonzero(
                ((pos - (mouse_x, mouse_y)) ** 2).sum(axis=1) < SMOOTHING_RADIUS ** 2)
            if len(nearby):
                sample_point = np.array([mouse_x, mouse_y], dtype=np.float64)
                nearby_pos = pos[nearby]
                density_at_cursor = calculate_density(nearby_pos, sample_point,
                                                      SMOOTHING_RADIUS)
                speed_at_cursor = calculate_property(
                    np.linalg.norm(vel[nearby], axis=1), sample_point, nearby_pos,
                    SMOOTHING_RADIUS, densities[nearby])
                probe_text = (f"cursor: density {density_at_cursor * inv_rest_density:.2f}x rest"
                              f" | speed {speed_at_cursor:.0f} px/s")
            else:
                probe_text = "cursor: no fluid nearby"
            if state["latched"]:
                mode_text = f"LATCHED, holding {int(grabbed.sum())} (click to release)"
            elif state["grab"]:
                mode_text = f"grabbing {int(grabbed.sum())}"
            else:
                mode_text = "idle"
            help_panel = pygame.Surface((1040, 125), pygame.SRCALPHA)
            help_panel.fill((8, 11, 18, 190))
            screen.blit(help_panel, (16, 16))
            screen.blit(font.render("PBF FLUID", True, (235, 242, 255)), (30, 25))
            screen.blit(font.render(
                "Double-click: grab, click: release | hold SPACE: push | wheel: radius | G: gravity | R: reset | H: help",
                True, (170, 190, 220)), (30, 55))
            screen.blit(font.render(
                f"{particle_count:,} particles | {clock.get_fps():.0f} FPS | gravity "
                f"{'on' if state['gravity'] else 'off'} | {mode_text}",
                True, (115, 205, 230)), (30, 80))
            screen.blit(font.render(probe_text, True, (115, 205, 230)), (30, 105))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()


# =====================================================================
# WHERE THE FLUID EQUATIONS ARE
#
# Navier-Stokes (incompressible):
#     dv/dt + (v . grad) v  =  -(1/rho) grad p  +  nu * laplacian(v)  +  g
#     div v = 0
#   term                        where it lives in this code
#   ---------------------------------------------------------------------
#   g       gravity             integrate():  vel_y += gravity * dt
#   external force (mouse)      integrate():  the grab / push blocks
#   (v.grad)v  advection        implicit: particles carry their own velocity,
#                               pred = pos + vel * dt   (integrate)
#   -grad p / rho  pressure     solve_lambda() + compute_dp()/apply_dp(). It is
#                               done by moving positions (Position Based Fluids)
#                               instead of adding a pressure force.
#   nu * laplacian(v)  viscosity  xsph() - a smoothing of velocities that behaves
#                               like viscosity (not the literal Laplacian)
#   div v = 0  incompressible   the SOLVER_ITERS loop in step(): keeps density at
#                               the rest density, which is the same condition
#                               (continuity: d rho / dt = -rho * div v)
#
# Gradient:   grad_x / grad_y in solve_lambda() and compute_dp() - the gradient
#             of the smoothing kernel ("spiky" kernel). It gives the direction
#             and strength to push each pair of particles apart.
# Convergence / divergence:  there is no explicit div v calculation. Crowding
#             (flow converging) shows up as density_error > 0 in solve_lambda();
#             the push-apart in compute_dp() removes it, so velocity divergence
#             stays about zero.
# =====================================================================