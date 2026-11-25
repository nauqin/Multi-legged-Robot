# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "hugo_base"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfgCPG(PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "hugo_cpg"

        # --max_iterations 를 안 주면 이만큼 학습한다 (이어 학습이면 추가분).
        self.max_iterations = 3000
        self.save_interval = 100

        # residual 은 작은 보정에서 시작하는 게 맞다
        self.policy.init_noise_std = 0.5
        self.algorithm.desired_kl = 0.005

        # 탐색 노이즈 폭주 방지.
        # 2026-09-15 run 에서 엔트로피가 22 -> 63 (표준편차 0.5 -> 약 2.0) 으로
        # 커지면서 두 번 붕괴했다. residual 스케일이 작아 노이즈가 보상을 거의
        # 해치지 않으니 엔트로피 보너스만 표준편차를 키운다.
        # CPG 가 기본 동작을 이미 만들므로 탐색 장려는 필요 없다.
        self.algorithm.entropy_coef = 0.0

        # 정책 출력을 [-1, 1] 로 자른다. residual 이 스케일 x 1 을 넘지 않아야
        # prismatic 예산 계산 (최악의 경우 d_min 0.010 m) 이 성립한다.
        # play.py 도 같은 값을 쓴다.
        self.clip_actions = 1.0


@configclass
class PPORunnerCfgCPG2(PPORunnerCfgCPG):
    """CPG + 수치 IK + residual (2026-11).

    v1 과 다른 점
      - 관측이 늘었다 (gait 상태 6 + 전방향 명령) -> 신경망을 키운다. [32, 32] 는
        지형 7 종 x 걸음새 3 종 x 전방향 명령을 담기에 작다.
      - 관측 정규화를 켠다. 높이 스캔(m), 관절 속도(rad/s), 진동자(-1~1) 의 크기가 다르다.
    v1 에서 얻은 안정화 설정(entropy 0, clip 1.0, desired_kl 0.005, init std 0.5)은 유지.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "hugo_cpg2"
        self.num_steps_per_env = 24
        self.max_iterations = 3000
        self.save_interval = 100
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
