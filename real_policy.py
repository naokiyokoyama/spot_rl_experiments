import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy
from habitat_baselines.utils.common import batch_obs

try:
    import magnum as mn
    import quaternion

    magnum_imported = True
except:
    print("FAILED TO IMPORT MAGNUM. Place Env will not work.")
    magnum_imported = False

# Turn numpy observations into torch tensors for consumption by policy
def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class RealPolicy:
    def __init__(self, checkpoint_path, observation_space, action_space, device):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]

        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        config.freeze()

        self.policy = PointNavBaselinePolicy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)

        # Load trained weights into the policy
        self.policy.load_state_dict(
            {k[len("actor_critic.") :]: v for k, v in checkpoint["state_dict"].items()}
        )

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def act(self, observations):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )
        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions


class GazePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
        observation_space = SpaceDict(
            {
                "arm_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 320, 1), dtype=np.float32
                ),
                "arm_depth_bbox": spaces.Box(
                    low=0.0, high=1.0, shape=(240, 320, 1), dtype=np.float32
                ),
                "joint": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "is_holding": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        action_space = spaces.Box(-1.0, 1.0, (4,))
        super().__init__(checkpoint_path, observation_space, action_space, device)


class PlacePolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
        observation_space = SpaceDict(
            {
                "joint": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "obj_start_sensor": spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
            }
        )
        action_space = spaces.Box(-1.0, 1.0, (4,))
        super().__init__(checkpoint_path, observation_space, action_space, device)


class NavPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
        observation_space = SpaceDict(
            {
                "spot_left_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "spot_right_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "goal_heading": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                ),
                "target_point_goal_gps_and_compass_sensor": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (2,))
        super().__init__(checkpoint_path, observation_space, action_space, device)


if magnum_imported:

    def spot2habitat_transform(position, rotation):
        x, y, z = position.x, position.y, position.z
        qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w

        quat = quaternion.quaternion(qw, qx, qy, qz)
        rotation_matrix = mn.Quaternion(quat.imag, quat.real).to_matrix()
        rotation_matrix_fixed = (
            rotation_matrix
            @ mn.Matrix4.rotation(
                mn.Rad(-np.pi / 2.0), mn.Vector3(1.0, 0.0, 0.0)
            ).rotation()
        )
        translation = mn.Vector3(x, z, -y)

        quat_rotated = mn.Quaternion.from_matrix(rotation_matrix_fixed)
        quat_rotated.vector = mn.Vector3(
            quat_rotated.vector[0], quat_rotated.vector[2], -quat_rotated.vector[1]
        )
        rotation_matrix_fixed = quat_rotated.to_matrix()
        sim_transform = mn.Matrix4.from_(rotation_matrix_fixed, translation)

        return sim_transform

else:

    def spot2habitat_transform(*args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    gaze_policy = GazePolicy(
        "weights/bbox_mask_5thresh_autograsp_shortrange_seed1_36.pth",
        device="cpu",
    )
    gaze_policy.reset()
    observations = {
        "arm_depth": np.zeros([240, 320, 1], dtype=np.float32),
        "arm_depth_bbox": np.zeros([240, 320, 1], dtype=np.float32),
        "joint": np.zeros(4, dtype=np.float32),
        "is_holding": np.zeros(1, dtype=np.float32),
    }
    actions = gaze_policy.act(observations)
    print("actions:", actions)
