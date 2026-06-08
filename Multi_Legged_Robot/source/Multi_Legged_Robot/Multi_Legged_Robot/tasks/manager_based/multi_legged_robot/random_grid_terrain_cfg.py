# random_grid_terrain_cfg.py

"""Random-grid rough terrain configuration for Hugo hexapod.

This terrain keeps only the MeshRandomGridTerrainCfg from IsaacLab rough terrains.
It is intended to generate irregular bumpy terrain made of random-height grid cells.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen


HUGO_RANDOM_GRID_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=True,

    # Each sub-terrain patch size.
    # 8m x 8m is the same scale used by IsaacLab rough terrain preset.
    size=(8.0, 8.0),

    # Large outer border around the whole terrain.
    border_width=20.0,
    border_height=1.0,

    # Terrain curriculum grid.
    # num_rows controls difficulty levels.
    # num_cols controls terrain type columns.
    # Since we use only one terrain type, columns are variations of the same random-grid terrain.
    num_rows=10,
    num_cols=20,

    # These mainly affect height-field terrains.
    # MeshRandomGridTerrainCfg does not rely on horizontal_scale like HfRandomUniformTerrainCfg,
    # but keeping these values aligned with IsaacLab rough preset is safe.
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,

    # Visual coloring.
    # "height" makes height differences easier to see.
    # If you want plain gray like your screenshot, use "none".
    color_scheme="none",

    # Initial difficulty range.
    # Because curriculum=True, rows gradually increase difficulty.
    difficulty_range=(0.0, 1.0),

    # Cache recommended after you settle the parameters.
    # True makes repeated runs faster and deterministic with the same seed.
    use_cache=True,
    cache_dir="/tmp/isaaclab/hugo_random_grid_terrains",

    sub_terrains={
        "random_grid_rough": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,

            # Main parameter: cell size.
            # Smaller value -> more dense and noisy-looking bumps.
            # IsaacLab rough preset uses 0.45.
            grid_width=0.35,

            # Main parameter: height range.
            # IsaacLab rough preset uses (0.05, 0.20).
            # Start slightly easier, then increase later if needed.
            grid_height_range=(0.03, 0.14),

            # Flat spawn area at the center of each sub-terrain.
            # Too small => robot may spawn on bad terrain and fail immediately.
            platform_width=2.0,

            # If True, only plus-shaped paths are generated and the rest becomes holes.
            # For normal bumpy terrain, keep False.
            holes=False,
        ),
    },
)


HUGO_RANDOM_GRID_TERRAIN_IMPORTER_CFG = terrain_gen.TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=HUGO_RANDOM_GRID_TERRAINS_CFG,

    # Use generated sub-terrain origins for env placement.
    use_terrain_origins=True,

    # Start from easier rows.
    # None means initial envs can spawn up to the hardest row.
    # For first training, use 2 or 3.
    max_init_terrain_level=2,

    # env_spacing is not used when use_terrain_origins=True,
    # but keep it for safety if you later set use_terrain_origins=False.
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