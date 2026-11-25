import argparse, gymnasium as gym, torch
from isaaclab.app import add_launcher_args, launch_simulation
import Multi_Legged_Robot.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "Hugo-Hexapod-v0"
parser = argparse.ArgumentParser()
parser.add_argument("--flat", action="store_true", help="평지에서 확인")
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args = parser.parse_args()

cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = 16
cfg.events.push_robot = None
if args.flat:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    act = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    z0 = robot.data.root_pos_w.torch[:, 2].clone()
    falls = 0
    for t in range(301):
        _, _, term, _, _ = env.step(act)
        falls += int(term.sum())
        if t <= 60 and t % 5 == 0 or t % 50 == 0:
            dz = robot.data.root_pos_w.torch[:, 2] - z0
            g = robot.data.projected_gravity_b.torch
            tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
            vz = robot.data.root_lin_vel_w.torch[:, 2]
            print(f"t={t:3d} 높이변화 평균={dz.mean():+.3f} 최대={dz.max():+.3f} m (env {int(dz.argmax())})  "
                  f"위로 속도 최대={vz.max():+.2f} m/s  기울기 최대={tilt.max():5.1f}°  넘어짐={falls}", flush=True)
    env.close()
