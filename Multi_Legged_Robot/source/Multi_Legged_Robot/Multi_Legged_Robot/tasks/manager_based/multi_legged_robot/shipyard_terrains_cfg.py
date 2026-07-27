# shipyard_terrains_cfg.py

from __future__ import annotations

import math
import numpy as np
import trimesh
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, SubTerrainBaseCfg

from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg


# ============================================================
# Common map setting
# 4096 robots = 64 x 64 origins
# robot spacing = 3 m
# total map footprint = 192 m x 192 m
# ============================================================

NUM_ENVS = 4096
NUM_ROWS = 64
NUM_COLS = 64
ENV_SPACING = 3.0

SUB_TERRAIN_SIZE = (3.0, 3.0)
HORIZONTAL_SCALE = 0.05
VERTICAL_SCALE = 0.001


# ============================================================
# Mesh utility
# ============================================================

def _height_grid_to_mesh(
    z: np.ndarray,
    size: tuple[float, float],
) -> trimesh.Trimesh:
    """Convert a height grid z[x, y] into a trimesh surface."""
    nx, ny = z.shape

    xs = np.linspace(0.0, size[0], nx)
    ys = np.linspace(0.0, size[1], ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    vertices = np.stack([xx, yy, z], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    return mesh


def _make_grid(size: tuple[float, float], scale: float):
    nx = int(size[0] / scale) + 1
    ny = int(size[1] / scale) + 1

    xs = np.linspace(0.0, size[0], nx)
    ys = np.linspace(0.0, size[1], ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return xx, yy


def _terrain_origin_center(size: tuple[float, float], z_center: float = 0.0) -> np.ndarray:
    return np.array([size[0] / 2.0, size[1] / 2.0, z_center], dtype=np.float32)


# ============================================================
# 1. Flat terrain
# ============================================================

def make_flat_terrain_cfg() -> TerrainImporterCfg:
    """Flat terrain for basic walking policy training and validation."""

    return TerrainImporterCfg(
        prim_path="/World/ground/flat_terrain",
        terrain_type="generator",
        num_envs=NUM_ENVS,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=1,
            curriculum=False,
            size=SUB_TERRAIN_SIZE,
            border_width=0.0,
            num_rows=NUM_ROWS,
            num_cols=NUM_COLS,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            slope_threshold=None,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(
                    function=flat_terrain,
                    proportion=1.0,
                    size=SUB_TERRAIN_SIZE,
                ),
            },
        ),
    )


# ============================================================
# 2. Slope terrain
#    10, 15, 20, 25, 30 degree slope
# ============================================================

def slope_terrain(difficulty: float, cfg: "ShipSlopeTerrainCfg"):
    """Generate a single planar slope with exact degree control."""

    xx, yy = _make_grid(cfg.size, cfg.horizontal_scale)

    slope_rad = math.radians(cfg.slope_deg)
    z = math.tan(slope_rad) * xx

    mesh = _height_grid_to_mesh(z, cfg.size)

    z_center = math.tan(slope_rad) * (cfg.size[0] / 2.0)
    origin = _terrain_origin_center(cfg.size, z_center)

    return [mesh], origin


@configclass
class ShipSlopeTerrainCfg(SubTerrainBaseCfg):
    function = slope_terrain

    slope_deg: float = 10.0
    horizontal_scale: float = HORIZONTAL_SCALE
    size: tuple[float, float] = SUB_TERRAIN_SIZE
    proportion: float = 1.0


def make_slope_terrain_cfg() -> TerrainImporterCfg:
    """Slope terrain with 10, 15, 20, 25, 30 degree sub-terrains."""

    return TerrainImporterCfg(
        prim_path="/World/ground/slope_terrain",
        terrain_type="generator",
        num_envs=NUM_ENVS,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=2,
            curriculum=False,
            size=SUB_TERRAIN_SIZE,
            border_width=0.0,
            num_rows=NUM_ROWS,
            num_cols=NUM_COLS,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            slope_threshold=None,
            sub_terrains={
                "slope_10deg": ShipSlopeTerrainCfg(slope_deg=10.0, proportion=0.2),
                "slope_15deg": ShipSlopeTerrainCfg(slope_deg=15.0, proportion=0.2),
                "slope_20deg": ShipSlopeTerrainCfg(slope_deg=20.0, proportion=0.2),
                "slope_25deg": ShipSlopeTerrainCfg(slope_deg=25.0, proportion=0.2),
                "slope_30deg": ShipSlopeTerrainCfg(slope_deg=30.0, proportion=0.2),
            },
        ),
    )


# ============================================================
# 3. Rough terrain
#    Perlin-noise-based irregular terrain
#    ±10 mm, ±20 mm, ±30 mm
# ============================================================

def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(a, b, t):
    return a + t * (b - a)


def _perlin_noise_2d(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Simple 2D gradient Perlin noise."""

    rng = np.random.default_rng(seed)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = _fade(x - x0)
    sy = _fade(y - y0)

    def random_gradient(ix, iy):
        hashed = (ix * 1836311903) ^ (iy * 2971215073) ^ seed
        hashed = np.abs(hashed) % (2**32)
        angles = (hashed / (2**32)) * 2.0 * np.pi
        return np.cos(angles), np.sin(angles)

    def dot_grid_gradient(ix, iy, x, y):
        gx, gy = random_gradient(ix, iy)
        dx = x - ix
        dy = y - iy
        return dx * gx + dy * gy

    n00 = dot_grid_gradient(x0, y0, x, y)
    n10 = dot_grid_gradient(x1, y0, x, y)
    n01 = dot_grid_gradient(x0, y1, x, y)
    n11 = dot_grid_gradient(x1, y1, x, y)

    ix0 = _lerp(n00, n10, sx)
    ix1 = _lerp(n01, n11, sx)
    value = _lerp(ix0, ix1, sy)

    return value


def perlin_rough_terrain(difficulty: float, cfg: "PerlinRoughTerrainCfg"):
    """Generate rough terrain using Perlin noise."""

    xx, yy = _make_grid(cfg.size, cfg.horizontal_scale)

    # Perlin coordinate scale
    x = xx * cfg.frequency
    y = yy * cfg.frequency

    seed = cfg.seed + np.random.randint(0, 1_000_000)

    noise = np.zeros_like(xx)
    amplitude_sum = 0.0

    amp = 1.0
    freq_mul = 1.0

    for _ in range(cfg.octaves):
        noise += amp * _perlin_noise_2d(x * freq_mul, y * freq_mul, seed)
        amplitude_sum += amp
        amp *= 0.5
        freq_mul *= 2.0

    noise = noise / max(amplitude_sum, 1e-6)

    max_abs = np.max(np.abs(noise))
    if max_abs > 1e-6:
        noise = noise / max_abs

    z = noise * cfg.height_amplitude

    mesh = _height_grid_to_mesh(z, cfg.size)
    origin = _terrain_origin_center(cfg.size, 0.0)

    return [mesh], origin


@configclass
class PerlinRoughTerrainCfg(SubTerrainBaseCfg):
    function = perlin_rough_terrain

    height_amplitude: float = 0.01   # meter
    frequency: float = 1.5
    octaves: int = 4
    seed: int = 0
    horizontal_scale: float = HORIZONTAL_SCALE
    size: tuple[float, float] = SUB_TERRAIN_SIZE
    proportion: float = 1.0


def make_rough_terrain_cfg() -> TerrainImporterCfg:
    """Rough terrain: ±10 mm, ±20 mm, ±30 mm Perlin noise."""

    return TerrainImporterCfg(
        prim_path="/World/ground/rough_terrain",
        terrain_type="generator",
        num_envs=NUM_ENVS,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=3,
            curriculum=False,
            size=SUB_TERRAIN_SIZE,
            border_width=0.0,
            num_rows=NUM_ROWS,
            num_cols=NUM_COLS,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            slope_threshold=None,
            sub_terrains={
                "rough_10mm": PerlinRoughTerrainCfg(
                    height_amplitude=0.010,
                    frequency=1.5,
                    octaves=4,
                    seed=10,
                    proportion=1.0 / 3.0,
                ),
                "rough_20mm": PerlinRoughTerrainCfg(
                    height_amplitude=0.020,
                    frequency=1.5,
                    octaves=4,
                    seed=20,
                    proportion=1.0 / 3.0,
                ),
                "rough_30mm": PerlinRoughTerrainCfg(
                    height_amplitude=0.030,
                    frequency=1.5,
                    octaves=4,
                    seed=30,
                    proportion=1.0 / 3.0,
                ),
            },
        ),
    )


# ============================================================
# 4. Stairs terrain
#    step height: 5~15 cm
#    step width : 20~30 cm
# ============================================================

def stairs_terrain(difficulty: float, cfg: "ShipStairsTerrainCfg"):
    """Generate one-directional stairs."""

    xx, yy = _make_grid(cfg.size, cfg.horizontal_scale)

    step_height = cfg.step_height
    step_width = cfg.step_width

    z = np.floor(xx / step_width) * step_height

    # Put the first step at z = 0
    z -= np.min(z)

    mesh = _height_grid_to_mesh(z, cfg.size)
    origin = _terrain_origin_center(cfg.size, float(z[z.shape[0] // 2, z.shape[1] // 2]))

    return [mesh], origin


@configclass
class ShipStairsTerrainCfg(SubTerrainBaseCfg):
    function = stairs_terrain

    step_height: float = 0.05
    step_width: float = 0.20
    horizontal_scale: float = HORIZONTAL_SCALE
    size: tuple[float, float] = SUB_TERRAIN_SIZE
    proportion: float = 1.0


def make_stairs_terrain_cfg() -> TerrainImporterCfg:
    """Stair terrain with heights 5~15 cm and widths 20~30 cm."""

    return TerrainImporterCfg(
        prim_path="/World/ground/stairs_terrain",
        terrain_type="generator",
        num_envs=NUM_ENVS,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=4,
            curriculum=False,
            size=SUB_TERRAIN_SIZE,
            border_width=0.0,
            num_rows=NUM_ROWS,
            num_cols=NUM_COLS,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            slope_threshold=None,
            sub_terrains={
                "stairs_h05_w20": ShipStairsTerrainCfg(
                    step_height=0.05,
                    step_width=0.20,
                    proportion=1.0 / 6.0,
                ),
                "stairs_h10_w25": ShipStairsTerrainCfg(
                    step_height=0.10,
                    step_width=0.25,
                    proportion=1.0 / 6.0,
                ),
                "stairs_h15_w30": ShipStairsTerrainCfg(
                    step_height=0.15,
                    step_width=0.30,
                    proportion=1.0 / 6.0,
                ),
                "stairs_h05_w30": ShipStairsTerrainCfg(
                    step_height=0.05,
                    step_width=0.30,
                    proportion=1.0 / 6.0,
                ),
                "stairs_h10_w20": ShipStairsTerrainCfg(
                    step_height=0.10,
                    step_width=0.20,
                    proportion=1.0 / 6.0,
                ),
                "stairs_h15_w25": ShipStairsTerrainCfg(
                    step_height=0.15,
                    step_width=0.25,
                    proportion=1.0 / 6.0,
                ),
            },
        ),
    )


# ============================================================
# 5. Curved metal surface
#    ship hull-like cylindrical / spherical metal surface
#    radius: 1~10 m
# ============================================================

def curved_metal_terrain(difficulty: float, cfg: "CurvedMetalTerrainCfg"):
    """Generate cylindrical or spherical curved surface."""

    xx, yy = _make_grid(cfg.size, cfg.horizontal_scale)

    x_centered = xx - cfg.size[0] / 2.0
    y_centered = yy - cfg.size[1] / 2.0

    radius = cfg.radius

    if cfg.surface_type == "cylindrical":
        # Arc length based cylinder surface.
        theta = x_centered / radius
        x_world = radius * np.sin(theta) + cfg.size[0] / 2.0
        y_world = yy
        z = radius * (1.0 - np.cos(theta))

    elif cfg.surface_type == "spherical":
        theta_x = x_centered / radius
        theta_y = y_centered / radius

        x_world = radius * np.sin(theta_x) + cfg.size[0] / 2.0
        y_world = radius * np.sin(theta_y) + cfg.size[1] / 2.0
        z = radius * (1.0 - np.cos(theta_x) * np.cos(theta_y))

    else:
        raise ValueError(f"Unknown surface_type: {cfg.surface_type}")

    vertices = np.stack([x_world, y_world, z], axis=-1).reshape(-1, 3)

    nx, ny = z.shape
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)

    origin = _terrain_origin_center(
        cfg.size,
        float(z[z.shape[0] // 2, z.shape[1] // 2]),
    )

    return [mesh], origin


@configclass
class CurvedMetalTerrainCfg(SubTerrainBaseCfg):
    function = curved_metal_terrain

    radius: float = 5.0
    surface_type: str = "cylindrical"  # "cylindrical" or "spherical"
    horizontal_scale: float = HORIZONTAL_SCALE
    size: tuple[float, float] = SUB_TERRAIN_SIZE
    proportion: float = 1.0


def make_curved_metal_terrain_cfg() -> TerrainImporterCfg:
    """Curved metal surface with radius 1~10 m."""

    return TerrainImporterCfg(
        prim_path="/World/ground/curved_metal_terrain",
        terrain_type="generator",
        num_envs=NUM_ENVS,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=5,
            curriculum=False,
            size=SUB_TERRAIN_SIZE,
            border_width=0.0,
            num_rows=NUM_ROWS,
            num_cols=NUM_COLS,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            slope_threshold=None,
            sub_terrains={
                "cylinder_r01": CurvedMetalTerrainCfg(
                    radius=1.0,
                    surface_type="cylindrical",
                    proportion=0.125,
                ),
                "cylinder_r03": CurvedMetalTerrainCfg(
                    radius=3.0,
                    surface_type="cylindrical",
                    proportion=0.125,
                ),
                "cylinder_r05": CurvedMetalTerrainCfg(
                    radius=5.0,
                    surface_type="cylindrical",
                    proportion=0.125,
                ),
                "cylinder_r10": CurvedMetalTerrainCfg(
                    radius=10.0,
                    surface_type="cylindrical",
                    proportion=0.125,
                ),
                "sphere_r01": CurvedMetalTerrainCfg(
                    radius=1.0,
                    surface_type="spherical",
                    proportion=0.125,
                ),
                "sphere_r03": CurvedMetalTerrainCfg(
                    radius=3.0,
                    surface_type="spherical",
                    proportion=0.125,
                ),
                "sphere_r05": CurvedMetalTerrainCfg(
                    radius=5.0,
                    surface_type="spherical",
                    proportion=0.125,
                ),
                "sphere_r10": CurvedMetalTerrainCfg(
                    radius=10.0,
                    surface_type="spherical",
                    proportion=0.125,
                ),
            },
        ),
    )


# ============================================================
# 6. Vertical metal surface
#    90-degree vertical wall for magnetic adhesion walking
# ============================================================

def vertical_surface_terrain(difficulty: float, cfg: "VerticalSurfaceTerrainCfg"):
    """Generate a thin vertical metal wall as a collision mesh."""

    thickness = cfg.thickness
    width = cfg.size[1]
    height = cfg.size[0]

    # Box extents: x thickness, y width, z height
    box = trimesh.creation.box(extents=(thickness, width, height))

    # Move box so that wall lies near x=0 and spans y,z positively.
    box.apply_translation(
        np.array(
            [
                thickness / 2.0,
                width / 2.0,
                height / 2.0,
            ]
        )
    )

    # Origin is on the surface center.
    origin = np.array(
        [
            thickness + cfg.surface_offset,
            width / 2.0,
            height / 2.0,
        ],
        dtype=np.float32,
    )

    return [box], origin


@configclass
class VerticalSurfaceTerrainCfg(SubTerrainBaseCfg):
    function = vertical_surface_terrain

    thickness: float = 0.05
    surface_offset: float = 0.02
    size: tuple[float, float] = SUB_TERRAIN_SIZE
    proportion: float = 1.0


def make_vertical_surface_terrain_cfg() -> TerrainImporterCfg:
    """Vertical metal wall terrain."""

    return TerrainImporterCfg(
        prim_path="/World/ground/vertical_surface_terrain",
        terrain_type="generator",
        num_envs=NUM_ENVS,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=6,
            curriculum=False,
            size=SUB_TERRAIN_SIZE,
            border_width=0.0,
            num_rows=NUM_ROWS,
            num_cols=NUM_COLS,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            slope_threshold=None,
            sub_terrains={
                "vertical_wall": VerticalSurfaceTerrainCfg(
                    thickness=0.05,
                    surface_offset=0.02,
                    proportion=1.0,
                ),
            },
        ),
    )


# ============================================================
# 7. Mixed terrain
#    Random course combining all terrains above
# ============================================================

def make_mixed_terrain_cfg() -> TerrainImporterCfg:
    """Mixed random terrain composed of flat, slope, rough, stairs, curved, and vertical surfaces."""

    return TerrainImporterCfg(
        prim_path="/World/ground/mixed_terrain",
        terrain_type="generator",
        num_envs=NUM_ENVS,
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=7,
            curriculum=False,
            size=SUB_TERRAIN_SIZE,
            border_width=0.0,
            num_rows=NUM_ROWS,
            num_cols=NUM_COLS,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            slope_threshold=None,
            sub_terrains={
                # Flat
                "flat": MeshPlaneTerrainCfg(
                    function=flat_terrain,
                    proportion=0.15,
                    size=SUB_TERRAIN_SIZE,
                ),

                # Slope
                "slope_10deg": ShipSlopeTerrainCfg(slope_deg=10.0, proportion=0.05),
                "slope_15deg": ShipSlopeTerrainCfg(slope_deg=15.0, proportion=0.05),
                "slope_20deg": ShipSlopeTerrainCfg(slope_deg=20.0, proportion=0.05),
                "slope_25deg": ShipSlopeTerrainCfg(slope_deg=25.0, proportion=0.05),
                "slope_30deg": ShipSlopeTerrainCfg(slope_deg=30.0, proportion=0.05),

                # Rough
                "rough_10mm": PerlinRoughTerrainCfg(
                    height_amplitude=0.010,
                    seed=101,
                    proportion=0.08,
                ),
                "rough_20mm": PerlinRoughTerrainCfg(
                    height_amplitude=0.020,
                    seed=102,
                    proportion=0.08,
                ),
                "rough_30mm": PerlinRoughTerrainCfg(
                    height_amplitude=0.030,
                    seed=103,
                    proportion=0.08,
                ),

                # Stairs
                "stairs_low": ShipStairsTerrainCfg(
                    step_height=0.05,
                    step_width=0.20,
                    proportion=0.08,
                ),
                "stairs_mid": ShipStairsTerrainCfg(
                    step_height=0.10,
                    step_width=0.25,
                    proportion=0.08,
                ),
                "stairs_high": ShipStairsTerrainCfg(
                    step_height=0.15,
                    step_width=0.30,
                    proportion=0.08,
                ),

                # Curved metal
                "cylindrical_metal_r03": CurvedMetalTerrainCfg(
                    radius=3.0,
                    surface_type="cylindrical",
                    proportion=0.04,
                ),
                "cylindrical_metal_r10": CurvedMetalTerrainCfg(
                    radius=10.0,
                    surface_type="cylindrical",
                    proportion=0.04,
                ),
                "spherical_metal_r05": CurvedMetalTerrainCfg(
                    radius=5.0,
                    surface_type="spherical",
                    proportion=0.04,
                ),

                # Vertical wall
                # 주의: 수직면은 robot spawn orientation을 별도로 90도 회전시켜야 합니다.
                "vertical_wall": VerticalSurfaceTerrainCfg(
                    thickness=0.05,
                    surface_offset=0.02,
                    proportion=0.05,
                ),
            },
        ),
    )