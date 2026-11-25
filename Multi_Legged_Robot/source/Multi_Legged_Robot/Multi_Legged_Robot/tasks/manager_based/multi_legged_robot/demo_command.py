"""발표 영상용 시나리오 속도 명령.

에피소드 시간에 따라 정해진 (vx, vy, wz, gait) 를 낸다. 모든 env 가 같은 시나리오를
같은 시각에 따라가므로 여러 지형을 나란히 찍기 좋다.

gait 칸: "auto" 면 상위 규칙(GaitSelector), 이름을 주면 그 gait 로 강제한다.
CPGIKAction 이 이 항의 gait_override 를 읽는다 (-1 = auto).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from .cpg_core import GAIT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# (시작 시각 s, vx, vy, wz, gait)
DEFAULT_SCHEDULE = (
    (0.0, 0.00, 0.00, 0.00, "auto"),     # 서기
    (3.0, 0.08, 0.00, 0.00, "auto"),     # 저속 -> wave
    (11.0, 0.16, 0.00, 0.00, "auto"),    # 중속 -> ripple
    (19.0, 0.30, 0.00, 0.00, "auto"),    # 고속 -> tripod
    (27.0, 0.00, 0.00, 0.30, "auto"),    # 제자리 좌회전
    (33.0, 0.00, 0.00, -0.30, "auto"),   # 제자리 우회전
    (39.0, 0.18, 0.00, 0.20, "auto"),    # 곡선 주행
    (47.0, 0.00, 0.12, 0.00, "auto"),    # 옆걸음 (좌)
    (53.0, 0.00, -0.12, 0.00, "auto"),   # 옆걸음 (우)
    (59.0, -0.15, 0.00, 0.00, "auto"),   # 후진
    (66.0, 0.00, 0.00, 0.00, "auto"),    # 정지
    (70.0, 0.00, 0.00, 0.00, "auto"),    # 끝
)


class ScriptedVelocityCommand(UniformVelocityCommand):
    cfg: "ScriptedVelocityCommandCfg"

    def __init__(self, cfg: "ScriptedVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        sch = cfg.schedule
        self._t0 = torch.tensor([s[0] for s in sch], device=self.device)
        self._vel = torch.tensor([[s[1], s[2], s[3]] for s in sch], device=self.device)
        self._gait = torch.tensor(
            [-1 if s[4] == "auto" else GAIT_NAMES.index(s[4]) for s in sch],
            dtype=torch.long, device=self.device)
        self.gait_override = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.segment = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        # 시나리오가 명령을 정하므로 무작위 재샘플링은 하지 않는다.
        pass

    def _update_command(self):
        t = self._env.episode_length_buf.float() * self._env.step_dt
        k = torch.searchsorted(self._t0, t, right=True) - 1
        k = k.clamp(0, len(self._t0) - 1)
        self.segment[:] = k
        self.vel_command_b[:] = self._vel[k]
        self.gait_override[:] = self._gait[k]


@configclass
class ScriptedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ScriptedVelocityCommand
    schedule: tuple = DEFAULT_SCHEDULE
