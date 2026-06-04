# preview_shipyard_terrain.py

from __future__ import annotations

import argparse


# ------------------------------------------------------------
# Isaac Sim / Isaac Lab AppLauncher
# AppLauncher는 Isaac Lab 관련 import보다 먼저 실행하는 것이 안전합니다.
# ------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Preview shipyard terrains in Isaac Sim."
)

parser.add_argument(
    "terrain_id",
    type=int,
    choices=[1, 2, 3, 4, 5, 6, 7],
    help=(
        "Terrain number: "
        "1=Flat, 2=Slope, 3=Rough, 4=Stairs, "
        "5=Curved metal, 6=Vertical surface, 7=Mixed"
    ),
)

parser.add_argument(
    "--small",
    action="store_true",
    help="Use a smaller 8x8 terrain grid for fast preview instead of full 64x64.",
)

parser.add_argument(
    "--rows",
    type=int,
    default=8,
    help="Number of terrain rows when --small is used.",
)

parser.add_argument(
    "--cols",
    type=int,
    default=8,
    help="Number of terrain columns when --small is used.",
)

# Isaac Lab AppLauncher arguments, e.g. --headless
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ------------------------------------------------------------
# Isaac Lab imports
# ------------------------------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter

from shipyard_terrains_cfg import (
    make_flat_terrain_cfg,
    make_slope_terrain_cfg,
    make_rough_terrain_cfg,
    make_stairs_terrain_cfg,
    make_curved_metal_terrain_cfg,
    make_vertical_surface_terrain_cfg,
    make_mixed_terrain_cfg,
)


TERRAIN_TABLE = {
    1: ("Flat terrain", make_flat_terrain_cfg),
    2: ("Slope terrain", make_slope_terrain_cfg),
    3: ("Rough terrain", make_rough_terrain_cfg),
    4: ("Stairs terrain", make_stairs_terrain_cfg),
    5: ("Curved metal surface", make_curved_metal_terrain_cfg),
    6: ("Vertical surface", make_vertical_surface_terrain_cfg),
    7: ("Mixed terrain", make_mixed_terrain_cfg),
}


def apply_small_preview(cfg, rows: int, cols: int):
    """Reduce terrain grid size for quick visual checking."""

    if cfg.terrain_generator is not None:
        cfg.terrain_generator.num_rows = rows
        cfg.terrain_generator.num_cols = cols

    cfg.num_envs = rows * cols
    return cfg


def main():
    terrain_name, terrain_cfg_func = TERRAIN_TABLE[args_cli.terrain_id]

    print("=" * 80)
    print(f"[Terrain Preview] Selected terrain {args_cli.terrain_id}: {terrain_name}")
    print("=" * 80)

    terrain_cfg = terrain_cfg_func()

    if args_cli.small:
        terrain_cfg = apply_small_preview(
            terrain_cfg,
            rows=args_cli.rows,
            cols=args_cli.cols,
        )
        print(f"[Preview Mode] Small terrain grid: {args_cli.rows} x {args_cli.cols}")
    else:
        print("[Preview Mode] Full terrain grid: 64 x 64")

    # ------------------------------------------------------------
    # Create simulation context
    # ------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # ------------------------------------------------------------
    # Light
    # ------------------------------------------------------------
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )
    light_cfg.func("/World/DomeLight", light_cfg)

    # ------------------------------------------------------------
    # Import terrain
    # TerrainImporter는 TerrainImporterCfg를 받아 지형을 World에 생성합니다.
    # ------------------------------------------------------------
    terrain_importer = TerrainImporter(terrain_cfg)

    # ------------------------------------------------------------
    # Camera view
    # ------------------------------------------------------------
    if args_cli.small:
        eye = [12.0, 12.0, 10.0]
        target = [6.0, 6.0, 0.0]
    else:
        eye = [95.0, 95.0, 85.0]
        target = [95.0, 95.0, 0.0]

    sim.set_camera_view(eye=eye, target=target)

    # ------------------------------------------------------------
    # Reset and run
    # ------------------------------------------------------------
    sim.reset()

    print("[INFO] Isaac Sim is running.")
    print("[INFO] Close the Isaac Sim window or press Ctrl+C in terminal to exit.")

    while simulation_app.is_running():
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()