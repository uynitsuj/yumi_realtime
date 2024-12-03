from yumi_realtime.controller import YuMiROSInterface
from loguru import logger
import numpy as onp
import tyro
import rospy
from typing import Literal
from yumi_realtime.base import YuMiBaseInterface
from yumi_realtime.policy_controller.utils.utils import *
# from dp_gs.dataset.utils import *
from dp_gs.policy.diffusion_wrapper import DiffusionWrapper
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import deque
from scipy.spatial.transform import Rotation
import PIL.Image as PILimage
import torch
import time

class YuMiDiffusionPolicyController(YuMiROSInterface):
    """YuMi controller for diffusion policy control."""
    
    def __init__(self, ckpt_path: str = None, ckpt_id: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interactive_handles = False
        
        for side in ['left', 'right']:
            target_handle = self.transform_handles[side]
            target_handle.control.visible = False
        
        assert ckpt_path is not None, "Diffusion Policy checkpoint path must be provided."
        
        # Setup Diffusion Policy module and weights
        self.model = DiffusionWrapper(model_ckpt_folder=ckpt_path, ckpt_id=ckpt_id, device='cuda')
        
        # ROS Camera Observation Subscriber
        self.height = None
        self.width = None
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        self.proprio_buffer = deque([],maxlen=self.model.model.obs_horizon)
        self.image_primary, self.image_wrist = deque([],maxlen=self.model.model.obs_horizon), deque([],maxlen=self.model.model.obs_horizon)
        self.action_queue = deque([],maxlen=self.model.model.action_horizon)
        self.prev_action = deque([],maxlen=self.model.model.obs_horizon)
        self.cur_proprio = None
        
        self.cartesian_pose_L = None
        self.cartesian_pose_R = None
        
        self.bridge = CvBridge()
        
        logger.info("Diffusion Policy controller initialized")
    
    def run(self):
        """Diffusion Policy controller loop."""
        rate = rospy.Rate(150) # 150Hz control loop          
        self.home()
        i = 0
        while ((self.height is None or self.width is None) or (self.cartesian_pose_L is None or self.cartesian_pose_R is None)):
            self.home()
            rate.sleep()
            self.solve_ik()
            self.update_visualization()
            
            super().publish_joint_commands()
            if i % 750 == 0:
                self.call_gripper(side = 'left', gripper_state = False, enable = True)
                self.call_gripper(side = 'right', gripper_state = False, enable = True)
                logger.info("Waiting for camera topic or robot pose data...")
                i = 0
            i += 1
        
        rospy.sleep(1.5)
        step = 0
        output_dir = "/home/xi/yumi_realtime/debugging_output"
        while not rospy.is_shutdown():
            input = {
            "observation": torch.from_numpy(onp.array(self.image_primary)).unsqueeze(0).unsqueeze(2), # [B, T, C, N_C, H, W]
            "proprio": torch.from_numpy(onp.array(self.proprio_buffer)).unsqueeze(0) # [B, T, 1, D] # TODO remove the unsqueeze(2) from proprio
                }
                    
            start = time.time()
            # action_prediction = self.model(input, denormalize=False) # Denoise action prediction from obs and proprio...
            action_prediction = self.model(input) # Denoise action prediction from obs and proprio...
            # action_prediction [B, T, D]
            
            # out_dict = {
            #     "observaiton": input["observation"].numpy(),
            #     "proprio": input["proprio"].numpy(),
            #     "action": action_prediction,
            # }
            # onp.save(f"{output_dir}/output_{step}.npy", out_dict)
            # step += 1

            # print("action: ", action_prediction[0, 0, :10])
            print("freq: ", 1/(time.time() - start))
            # start = time.time()
            
            # action_L = convert_abs_action(action_prediction[:,:,:10],self.cur_proprio[:10][None,None])[0]
            # action_R = convert_abs_action(action_prediction[:,:,10:],self.cur_proprio[10:][None,None])[0]
            action_L = action_prediction[0,:,:10]
            action_R = action_prediction[0,:,10:]
            
            action = onp.concatenate([action_L, action_R], axis=-1)
            
            # # only first action
            # action = action[0]
            
            # # temporal emsemble start
            # new_actions = deque(action[:self.model.model.action_horizon])
            # self.action_queue.append(new_actions)
            # actions_current_timestep = onp.empty((len(self.action_queue), self.model.model.action_dim))
            
            # k = 0.05
            # for i, q in enumerate(self.action_queue):
            #     actions_current_timestep[i] = q.popleft()
            # exp_weights = onp.exp(k * onp.arange(actions_current_timestep.shape[0]))
            # exp_weights = exp_weights / exp_weights.sum()
            # action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            
            # receeding horizon 
            if len(self.action_queue) == 0: 
                # self.action_queue = deque([a for a in action[:self.model.model.action_horizon]]) 
                self.action_queue = deque([a for a in action[:10]]) 
            action = self.action_queue.popleft()
            
            # YuMi action update
            ######################################################################
            l_act = action_10d_to_8d(action[:10])
            r_act = action_10d_to_8d(action[10:])
            l_xyz, l_wxyz, l_gripper_cmd = l_act[:3], l_act[3:-1], l_act[-1]
            r_xyz, r_wxyz, r_gripper_cmd = r_act[:3], r_act[3:-1], r_act[-1]
            print("left xyz: ", l_xyz)
            print("left gripper: ", l_gripper_cmd)
            
            super().update_target_pose(
            side='left',
            position=l_xyz,
            wxyz=l_wxyz,
            gripper_state=l_gripper_cmd, 
            enable=True
            )
            
            super().update_target_pose(
            side='right',
            position=r_xyz,
            wxyz=r_wxyz,
            gripper_state=r_gripper_cmd, 
            enable=True
            )
            ######################################################################
            
            self.solve_ik()
            self.update_visualization()
            super().publish_joint_commands()
            
            rate.sleep()
    
    def update_curr_proprio(self):
        l_xyz, l_xyzw = tf2xyz_quat(self.cartesian_pose_L)
        r_xyz, r_xyzw = tf2xyz_quat(self.cartesian_pose_R)
        
        l_q = Rotation.from_quat(l_xyzw)
        l_rot = l_q.as_matrix()
        l_rot_6d = onp.squeeze(rot_mat_to_rot_6d(l_rot[None]), axis=0)# [N, 6]
        r_q = Rotation.from_quat(r_xyzw)
        r_rot = r_q.as_matrix()
        r_rot_6d = onp.squeeze(rot_mat_to_rot_6d(r_rot[None]), axis=0) # [N, 6]
        
        self.cur_proprio = onp.concatenate([l_xyz, l_rot_6d, onp.array([int(self.gripper_L_pos)/10000]), r_xyz, r_rot_6d, onp.array([int(self.gripper_R_pos)/10000])], axis=-1, dtype=onp.float32)
        assert self.cur_proprio.shape == (20,)

        self.proprio_buffer.append(self.cur_proprio)
    
    def image_callback(self, image_msg: Image):
        """Handle camera observation updates."""
        if self.height is None and self.width is None:
            self.height = image_msg.height
            self.width = image_msg.width
            logger.info(f"First image received; Observation dim: {self.height}x{self.width}x3")
        
        onp_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8').astype("float32") / 255.0  # H, W, C
        
        new_obs = onp.transpose(onp_img, (2, 0, 1)) # C, H, W
        
        if self.cartesian_pose_L is None or self.cartesian_pose_R is None:
            return
        
        while len(self.image_primary) < self.model.model.obs_horizon - 1:
            self.image_primary.append(new_obs)
            self.update_curr_proprio()

        self.image_primary.append(new_obs)
        self.update_curr_proprio()
        
        # Ensure both buffers are full at this point
        assert len(self.image_primary) == self.model.model.obs_horizon
        assert len(self.proprio_buffer) == self.model.model.obs_horizon
              
    def episode_start(self):
        """Reset the environment and start a new episode."""
        self.image_primary.clear()
        self.image_wrist.clear()
        self.prev_action.clear()
        self.proprio_buffer.clear()
        self.action_queue.clear()
        self.cur_proprio = None
        self.home() # Move to home position as first action
        rospy.sleep(5)
        
    def plot_action_queue(self, side: Literal['left', 'right'] = None, color: Tuple = (255,0,0), size: int = 5):
        """Plot the action queue for the given side."""
        
        for i, action in enumerate(self.action_queue):
            if side == 'left':
                self.plot_action(action[:10], color, size)
            elif side == 'right':
                self.plot_action(action[10:], color, size)
            
    
def main(
    # ckpt_path: str = "/home/xi/checkpoints/241122_1324",
    # ckpt_path: str = "/home/xi/checkpoints/simplepolicy_241122_1526",
    # ckpt_path: str = "/home/xi/checkpoints/simple_policy_241122_1644",
    # ckpt_path: str = "/home/xi/checkpoints/241124_2117",
    # ckpt_path: str = "/home/xi/checkpoints/241125_2130",
    # ckpt_path: str = "/home/xi/checkpoints/241126_1511",
    # ckpt_path: str = "/home/xi/checkpoints/241126_1646",
    # ckpt_path: str = "/home/xi/checkpoints/241126_1727",
    # ckpt_path: str = "/home/xi/checkpoints/241127_1049",
    # ckpt_path: str = "/home/xi/checkpoints/241202_1331",
    ckpt_path: str = "/home/xi/checkpoints/241202_2333",
    # ckpt_path: str = "/home/xi/checkpoints/241202_2334",
    ckpt_id: int = 60
    ): 
    
    yumi_interface = YuMiDiffusionPolicyController(ckpt_path, ckpt_id)
    yumi_interface.run()
    
if __name__ == "__main__":
    tyro.cli(main)
