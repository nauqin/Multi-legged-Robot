import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


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