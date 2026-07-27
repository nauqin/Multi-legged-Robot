import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

GRAVITY_MAG = 9.81

def feet_contact_state(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = (
        contact_sensor.data.net_forces_w_history[
            :, :, sensor_cfg.body_ids, :
        ]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    

    contacts = forces > threshold
    return contacts.float()

def feet_contact_force(
    env,
    sensor_cfg: SceneEntityCfg,
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = (
        contact_sensor.data.net_forces_w_history[
            :, :, sensor_cfg.body_ids, :
        ]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    

    return torch.clamp(forces / 1000.0, 0.0, 1.0)

##
# IMU observation functions
##

def imu_ang_vel_b(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """IMU angular velocity in body frame."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.ang_vel_b


def imu_projected_gravity_b(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """Gravity direction projected into body frame."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.projected_gravity_b


def imu_lin_acc_residual_b(
    env,
    sensor_cfg: SceneEntityCfg,
    gravity: float = GRAVITY_MAG,
):
    """Body-frame acceleration with gravity removed.

    Useful for:
    - landing impact detection
    - body vibration estimation
    - vertical oscillation estimation
    """

    imu = env.scene[sensor_cfg.name]

    residual_acc_b = (
        imu.data.lin_acc_b
        + gravity * imu.data.projected_gravity_b
    )

    return residual_acc_b / gravity