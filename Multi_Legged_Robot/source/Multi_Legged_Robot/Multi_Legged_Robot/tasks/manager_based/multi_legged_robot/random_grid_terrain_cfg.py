# random_grid_terrain_cfg.py

"""Mixed rough terrain configuration for Hugo hexapod.

Terrain composition:
- Flat terrain
- Random-grid bumpy terrain
- Pyramid sloped terrain
- Pyramid stairs terrain

You can adjust the terrain ratio by changing only:
    FLAT_RATIO
    RANDOM_GRID_RATIO
    SLOPE_RATIO
    STAIRS_RATIO

The terrain difficulty increases gradually along rows because curriculum=True.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen


##
# Terrain ratio settings
##

# Unit: percent-like weight.
# They do not have to sum exactly to 100.
# The code normalizes them internally.
#
# Recommended first setting:
#   flat 20%, random grid 30%, slope 30%, stairs 20%
FLAT_RATIO = 100.0
RANDOM_GRID_RATIO = 0.0
SLOPE_RATIO = 0.0
STAIRS_RATIO = 0.0


def _normalize_ratio(value: float, total: float) -> float:
    """Convert user-facing ratio value into TerrainGenerator proportion."""
    if total <= 0.0:
        raise ValueError(
            "The sum of terrain ratios must be positive. "
            "Please check FLAT_RATIO, RANDOM_GRID_RATIO, SLOPE_RATIO, STAIRS_RATIO."
        )
    return float(value) / float(total)


_TOTAL_RATIO = FLAT_RATIO + RANDOM_GRID_RATIO + SLOPE_RATIO + STAIRS_RATIO

FLAT_PROPORTION = _normalize_ratio(FLAT_RATIO, _TOTAL_RATIO)
RANDOM_GRID_PROPORTION = _normalize_ratio(RANDOM_GRID_RATIO, _TOTAL_RATIO)
SLOPE_PROPORTION = _normalize_ratio(SLOPE_RATIO, _TOTAL_RATIO)
STAIRS_PROPORTION = _normalize_ratio(STAIRS_RATIO, _TOTAL_RATIO)


##
# Global terrain generation settings
##

# Each sub-terrain patch is 8 m x 8 m, same as the IsaacLab rough terrain preset.
SUB_TERRAIN_SIZE = (8.0, 8.0)

# num_rows controls difficulty levels.
# With curriculum=True, row 0 is easy and later rows become harder.
NUM_ROWS = 64

# num_cols controls terrain-type distribution.
# Since 20 columns and 20/30/30/20 ratio:
#   flat        -> about 4 columns
#   random_grid -> about 6 columns
#   slope       -> about 6 columns
#   stairs      -> about 4 columns
NUM_COLS = 64

# Difficulty range used for curriculum.
# Smaller upper bound makes all terrains easier.
DIFFICULTY_RANGE = (0.0, 1.0)

# Terrain cache.
# If you change terrain parameters and want a fresh terrain, either:
#   1. change CACHE_DIR, or
#   2. delete the old cache directory.
USE_CACHE = True
CACHE_DIR = "/tmp/isaaclab/hugo_mixed_rough_terrains"


##
# Individual terrain difficulty settings
##

# 1. Flat terrain
# Flat terrain ignores difficulty, but it is useful for preventing catastrophic forgetting.
FLAT_SETTINGS = dict(
    proportion=FLAT_PROPORTION,
)

# 2. Random-grid bumpy terrain
#
# IsaacLab rough preset:
#   grid_width=0.45
#   grid_height_range=(0.05, 0.20)
#   platform_width=2.0
#
# Here:
#   grid_width=0.35 gives denser bumps.
#   grid_height_range=(0.03, 0.14) starts easier.
# Increase to (0.05, 0.20) later if the robot learns well.
RANDOM_GRID_SETTINGS = dict(
    proportion=RANDOM_GRID_PROPORTION,
    grid_width=0.35,
    grid_height_range=(0.03, 0.14),
    platform_width=2.0,
    holes=False,
)

# 3. Pyramid sloped terrain
#
# IsaacLab rough preset:
#   slope_range=(0.0, 0.4)
#   platform_width=2.0
#   border_width=0.25
#
# With curriculum=True, the effective slope gradually increases by row.
SLOPE_SETTINGS = dict(
    proportion=SLOPE_PROPORTION,
    slope_range=(0.0, 0.4),
    platform_width=2.0,
    border_width=0.25,
)

# 4. Pyramid stairs terrain
#
# IsaacLab rough preset:
#   step_height_range=(0.05, 0.23)
#   step_width=0.3
#   platform_width=3.0
#   border_width=1.0
#
# With curriculum=True, the effective step height gradually increases by row.
STAIRS_SETTINGS = dict(
    proportion=STAIRS_PROPORTION,
    step_height_range=(0.05, 0.23),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
    holes=False,
)


##
# Terrain generator config
##

HUGO_MIXED_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=True,

    # Sub-terrain patch size.
    size=SUB_TERRAIN_SIZE,

    # Large outer border around the full terrain.
    border_width=20.0,
    border_height=1.0,

    # Curriculum grid.
    num_rows=NUM_ROWS,
    num_cols=NUM_COLS,

    # Mainly used by height-field terrains such as HfPyramidSlopedTerrainCfg.
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,

    # "none": plain gray material.
    # "height": color by terrain height, useful for visual debugging.
    # "random": random vertex colors.
    color_scheme="none",

    difficulty_range=DIFFICULTY_RANGE,

    use_cache=USE_CACHE,
    cache_dir=CACHE_DIR,

    sub_terrains={
        # Flat terrain for maintaining basic walking ability.
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            **FLAT_SETTINGS,
        ),

        # Bumpy random-grid terrain.
        "random_grid_rough": terrain_gen.MeshRandomGridTerrainCfg(
            **RANDOM_GRID_SETTINGS,
        ),

        # Sloped terrain, close to IsaacLab rough preset.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            **SLOPE_SETTINGS,
        ),

        # Stairs terrain, close to IsaacLab rough preset.
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            **STAIRS_SETTINGS,
        ),
    },
)


##
# Terrain importer config
##

HUGO_MIXED_ROUGH_TERRAIN_IMPORTER_CFG = terrain_gen.TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=HUGO_MIXED_ROUGH_TERRAINS_CFG,

    # Use generated sub-terrain origins for environment placement.
    use_terrain_origins=True,

    # Start from easier rows.
    # 0 means only the easiest row at start.
    # 2 means rows 0~2 at start.
    # None means envs can start from all rows including the hardest.
    max_init_terrain_level=2,

    # Not used when use_terrain_origins=True, but kept for safety.
    env_spacing=4.0,

    debug_vis=False,

    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),

    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.75, 0.75, 0.75),
    ),
)


##
# Backward-compatible aliases
##

# These aliases allow the existing env_cfg code to keep using the old names.
# If your env_cfg imports HUGO_RANDOM_GRID_TERRAIN_IMPORTER_CFG,
# it will still work with this mixed terrain.
HUGO_RANDOM_GRID_TERRAINS_CFG = HUGO_MIXED_ROUGH_TERRAINS_CFG
HUGO_RANDOM_GRID_TERRAIN_IMPORTER_CFG = HUGO_MIXED_ROUGH_TERRAIN_IMPORTER_CFG