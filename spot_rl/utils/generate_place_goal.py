import magnum as mn
import numpy as np
from spot_wrapper.spot import Spot

from spot_rl.envs.base_env import SpotBaseEnv

EE_GRIPPER_OFFSET = [0.2, 0.0, 0.05]


def main(spot: Spot):
    position, rotation = spot.get_base_transform_to("link_wr1")
    wrist_T_base = SpotBaseEnv.spot2habitat_transform(position, rotation)
    gripper_T_base = wrist_T_base @ mn.Matrix4.translation(
        mn.Vector3(EE_GRIPPER_OFFSET)
    )
    base_place_target_habitat = np.array(gripper_T_base.translation)
    base_place_target = base_place_target_habitat[[0, 2, 1]]

    x, y, yaw = spot.get_xy_yaw()
    base_T_global = mn.Matrix4.from_(
        mn.Matrix4.rotation_z(mn.Rad(yaw)).rotation(),
        mn.Vector3(mn.Vector3(x, y, 0.5)),
    )
    global_place_target = base_T_global.transform_point(base_place_target)
    print(global_place_target)


if __name__ == "__main__":
    spot = Spot("PlaceGoalGenerator")
    main(spot)
