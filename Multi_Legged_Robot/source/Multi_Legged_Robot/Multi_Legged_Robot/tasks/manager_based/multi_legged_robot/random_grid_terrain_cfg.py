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

NOTE: curriculum=True only *generates* difficulty levels along rows.
      To actually promote/demote robots between levels you MUST also add
      a curriculum term in the env cfg:

          from isaaclab.managers import CurriculumTermCfg as CurrTerm

          @configclass
          class CurriculumCfg:
              terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

      and register it on the env cfg:

          curriculum: CurriculumCfg = CurriculumCfg()

      Without it, robots stay on their initial rows forever and rows
      above max_init_terrain_level are generated but never used.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen

##
# Terrain ratio settings
##

# Unit: percent-like weight.
# They do not have to sum exactly to 100.
# The code normalizes them internally.
#
# Stairs are the hardest terrain for a hexapod, and terrain_levels_vel
# promotes/demotes based on travelled distance. Too many stair columns
# early on pins the curriculum level near 0, so stairs start low here.
# Once the mean terrain level climbs steadily, raise STAIRS_RATIO.
FLAT_RATIO = 20.0
RANDOM_GRID_RATIO = 35.0
SLOPE_RATIO = 30.0
STAIRS_RATIO = 15.0


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
#
# num_rows  -> difficulty levels. With curriculum=True, row 0 is easiest and
#              the last row is hardest.
# num_cols  -> terrain-type distribution. Columns are assigned deterministically
#              by cumulative proportion.
#
# With NUM_COLS = 16 and 20 / 35 / 30 / 15:
#   flat        -> 3 columns
#   random_grid -> 6 columns
#   slope       -> 5 columns
#   stairs      -> 2 columns

# Difficulty range used for curriculum.
# Smaller upper bound makes all terrains easier.
DIFFICULTY_RANGE = (0.0, 1.0)

# Terrain cache.
# If you change terrain parameters and want a fresh terrain, either:
#   1. change CACHE_DIR, or
#   2. delete the old cache directory.
#
# Select the profile with the HUGO_TERRAIN_PROFILE environment variable:
#   HUGO_TERRAIN_PROFILE=play python scripts/rsl_rl/play.py ...
TERRAIN_PROFILE = os.environ.get("HUGO_TERRAIN_PROFILE", "train")

if TERRAIN_PROFILE == "train":
    NUM_ROWS = 8
    NUM_COLS = 16
    SUB_TERRAIN_SIZE = (8.0, 8.0)
    USE_CACHE = True
    CACHE_DIR = "/tmp/isaaclab/hugo_mixed_rough_terrains_train_v2"
elif TERRAIN_PROFILE == "play":
    NUM_ROWS = 8
    NUM_COLS = 16
    SUB_TERRAIN_SIZE = (8.0, 8.0)
    USE_CACHE = False
    CACHE_DIR = "/tmp/isaaclab/hugo_mixed_rough_terrains_play_v2"
else:
    raise ValueError(
        f"Unknown HUGO_TERRAIN_PROFILE: {TERRAIN_PROFILE!r}. Expected 'train' or 'play'."
    )

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
#   Increase to (0.05, 0.20) later if the robot learns well.
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
# The lower bound must stay near 0.0. With curriculum=True the effective slope
# is interpolated across this range by row, so a lower bound of 0.2 would make
# even the easiest row an 11-degree slope and throw away half the curriculum.
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
# 0.23 m steps are very tall for a hexapod. Start lower and raise the upper
# bound once the mean terrain level is climbing.
STAIRS_SETTINGS = dict(
    proportion=STAIRS_PROPORTION,
    step_height_range=(0.05, 0.16),
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
    # 0.0 keeps the border flush with the terrain. A raised border acts as a
    # wall that the robot can collide with and that the height scanner reads.
    border_height=0.0,
    # Curriculum grid.
    num_rows=NUM_ROWS,
    num_cols=NUM_COLS,
    # Mainly used by height-field terrains such as HfPyramidSlopedTerrainCfg.
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    # "none":   plain gray material.
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
    #   0    -> only the easiest row at start.
    #   2    -> rows 0~2 at start.
    #   None -> envs can start from all rows including the hardest.
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

# Force the play profile:
#   HUGO_TERRAIN_PROFILE=play python scripts/rsl_rl/play.py \
#       --task Hugo-Hexapod-v0 --num_envs 4 --checkpoint <checkpoint_path>
#
# Clear the cache:
#   rm -rf /tmp/isaaclab/hugo_mixed_rough_terrains_train_v2
#   rm -rf /tmp/isaaclab/hugo_mixed_rough_terrains_play_v2