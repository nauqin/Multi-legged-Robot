"""CPG2 속도 명령 항: 걸음새 속도 제한 (학습용) + 발표 영상용 시나리오.

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


def gait_command_scale(env, action_term_name: str, cmd: torch.Tensor):
    """CPG 액션 항에게 지금 gait 로 낼 수 있는 속도 배율을 묻는다. 항이 없으면 None."""
    try:
        term = env.action_manager.get_term(action_term_name)
    except Exception:  # noqa: BLE001  (환경 생성 중에는 action_manager 가 아직 없다)
        return None
    fn = getattr(term, "command_scale", None)
    return None if fn is None else fn(cmd)


class GaitLimitedVelocityCommand(UniformVelocityCommand):
    """무작위 속도 명령을 지금 걸음새가 낼 수 있는 속도 안으로 줄인다.

    걸음새마다 최대 속도가 다르다 (wave 0.14, ripple 0.25, tripod 0.39 m/s). 줄이지
    않으면 wave/ripple 로 걷는 동안이나 회전이 섞인 명령에서 CPG 가 낼 수 없는 속도를
    요구하게 되고, 정책은 잔차로 메울 수 없는 속도 추종 벌점을 계속 받는다.

    - 뽑힌 그대로의 명령은 unclipped_command 에 둔다. 걸음새 선택기는 이 값을 본다
      (줄인 명령을 보면 느린 걸음새에 갇힐 수 있다).
    - 관측, 보상, CPG 는 줄인 명령 (command) 을 쓴다.
    """

    cfg: "GaitLimitedVelocityCommandCfg"

    def __init__(self, cfg: "GaitLimitedVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._sampled = torch.zeros_like(self.vel_command_b)
        self.unclipped_command = torch.zeros_like(self.vel_command_b)
        self.command_scale = torch.ones(self.num_envs, device=self.device)
        self.metrics["cmd_scale"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["cmd_limited_time"] = torch.zeros(self.num_envs, device=self.device)
        self._scale_sum = torch.zeros(self.num_envs, device=self.device)
        self._limited_sum = torch.zeros(self.num_envs, device=self.device)
        self._scale_cnt = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        n = self._scale_cnt[ids].clamp_min(1.0)
        self.metrics["cmd_scale"][ids] = self._scale_sum[ids] / n
        self.metrics["cmd_limited_time"][ids] = self._limited_sum[ids] / n
        self._scale_sum[ids] = 0.0
        self._limited_sum[ids] = 0.0
        self._scale_cnt[ids] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self._sampled[env_ids] = self.vel_command_b[env_ids]
        # 리셋 직후 첫 걸음새 결정이 이 값을 바로 읽는다 (_update_command 는 그 뒤에 불린다).
        # 여기서 채워 두지 않으면 0 으로 읽혀 매 에피소드가 wave 로 시작한다.
        raw = self._sampled[env_ids].clone()
        raw[self.is_standing_env[env_ids]] = 0.0
        self.unclipped_command[env_ids] = raw

    def _update_command(self):
        # 매 스텝 뽑힌 값에서 다시 시작한다 (줄인 값을 또 줄이지 않도록)
        self.vel_command_b[:] = self._sampled
        super()._update_command()                      # 서 있는 env 는 0
        self.unclipped_command[:] = self.vel_command_b
        if not self.cfg.limit_to_gait:
            return
        scale = gait_command_scale(self._env, self.cfg.action_term_name, self.vel_command_b)
        if scale is None:
            return
        self.command_scale[:] = scale
        self.vel_command_b *= scale.unsqueeze(-1)
        self._scale_sum += scale
        self._limited_sum += (scale < 0.98).float()
        self._scale_cnt += 1.0


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
        self.unclipped_command = torch.zeros_like(self.vel_command_b)

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
        # 학습과 같은 조건: 지금 걸음새가 낼 수 있는 속도 안으로 줄인다
        self.unclipped_command[:] = self.vel_command_b
        if self.cfg.limit_to_gait:
            scale = gait_command_scale(self._env, self.cfg.action_term_name, self.vel_command_b)
            if scale is not None:
                self.vel_command_b *= scale.unsqueeze(-1)


@configclass
class GaitLimitedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GaitLimitedVelocityCommand
    limit_to_gait: bool = True
    action_term_name: str = "cpg"


@configclass
class ScriptedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ScriptedVelocityCommand
    schedule: tuple = DEFAULT_SCHEDULE
    limit_to_gait: bool = True
    action_term_name: str = "cpg"
