from yumi_realtime.controller import YuMiROSInterface
from loguru import logger
import numpy as onp
import tyro
from typing import Tuple
import rospy
from yumi_realtime.data_logging.data_collector import DataCollector
from yumi_realtime.base import YuMiBaseInterface
from yumi_realtime.policy_controller.utils.utils import *
from dp_gs.policy.diffusion_wrapper import DiffusionPolicyWrapper, normalize, unnormalize
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from collections import deque
import torch
from scipy.spatial.transform import Rotation

class YuMiDiffusionPolicyController(YuMiROSInterface):
    """YuMi controller for diffusion policy control."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interactive_handles = False
        
        # Setup Diffusion Policy module and weights
        self.model = DiffusionPolicyWrapper()
        
        # ROS Camera Observation Subscriber
        self.height = None
        self.width = None
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        self.proprio_buffer = deque([],maxlen=self.model.obs_horizon)
        self.image_primary, self.image_wrist = deque([],maxlen=self.model.obs_horizon), deque([],maxlen=self.model.obs_horizon)
        self.action_queue = deque([],maxlen=self.model.action_horizon)
        self.prev_action = deque([],maxlen=self.model.obs_horizon)
        self.cur_proprio = None
        
        self.bridge = CvBridge()
        
        logger.info("Diffusion Policy controller initialized")
    
    def run(self):
        """Diffusion Policy controller loop."""
        rate = rospy.Rate(250) # 250Hz control loop          
        
        while ((self.height is None or self.width is None) or (self.cartesian_pose_L is None or self.cartesian_pose_R is None)):
            self.home() # Move to home position as first action
            rospy.sleep(5)
            rate.sleep() # Wait for first inputs to arrive
        
        while not rospy.is_shutdown():
            input = {
            "observation": onp.array(self.image_primary),
            "proprio": onp.array(self.proprio_buffer)
                }
        
            action_prediction = self.model.forward(input) # Denoise action prediction from obs and proprio...
            
            action = convert_abs_action(action_prediction[None],self.cur_proprio[None,None])[0] # action_horizon, action_dim
            # temporal emsemble start
            new_actions = deque(action[:self.model.action_horizon])
            self.action_queue.append(new_actions)
            actions_current_timestep = onp.empty((len(self.action_queue), self.model.action_dim))
            
            k = 0.05
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()
            exp_weights = onp.exp(k * onp.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()
            action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            
            # TODO: Convert denoised action_prediction format to commands for YuMi
            ######################################################################
            l_xyz, l_wxyz = None, None
            r_xyz, r_wxyz = None, None
            
            super().update_target_pose(
            side='left',
            position=l_xyz,
            wxyz=l_wxyz,
            gripper_state=False, # Binary
            enable=True
            )
            
            super().update_target_pose(
            side='right',
            position=r_xyz,
            wxyz=r_wxyz,
            gripper_state=False, # Binary
            enable=True
            )
            ######################################################################
            
            YuMiBaseInterface.solve_ik()
            YuMiBaseInterface.update_visualization()
            super().publish_joint_commands()
            
            rate.sleep()
    
    def update_curr_proprio(self):
        l_xyz, l_xyzw = tf2xyz_quat(self.cartesian_pose_L)
        r_xyz, r_xyzw = tf2xyz_quat(self.cartesian_pose_R)
        
        l_q = Rotation.from_quat(l_xyzw)
        l_rot = l_q.as_matrix()
        l_rot_6d = rot_mat_to_rot_6d(l_rot) # [N, 6]
        r_q = Rotation.from_quat(r_xyzw)
        r_rot = r_q.as_matrix()
        r_rot_6d = rot_mat_to_rot_6d(r_rot) # [N, 6]
        
        self.cur_proprio = onp.concatenate([l_xyz, l_rot_6d, int(self.gripper_L_pos.value)/10000, r_xyz, r_rot_6d, int(self.gripper_R_pos.value)/10000], axis=-1)
        self.proprio_buffer.append(self.cur_proprio)
        
    def image_callback(self, image_msg: Image):
        """Handle camera observation updates."""
        if self.height is None and self.width is None:
            self.height = image_msg.height
            self.width = image_msg.width
            logger.info(f"First image received; Observation dim: {self.height}x{self.width}x3")
            
        #todo: center crop image or padding    
        image_msg  = image_msg.resize((224, 224), Image.Resampling.LANCZOS)
        
        onp_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8').astype("float32") / 255.0  # Normalized to float [0, 1]
        
        new_obs = onp_img.permute(2, 0, 1)
        self.image_primary.append(new_obs)
        self.update_curr_proprio()

    def _control_l(self, data):
        """Handle left controller updates."""
        l_wxyz = onp.array([
            data.target_cartesian_pos.transform.rotation.w,
            data.target_cartesian_pos.transform.rotation.x,
            data.target_cartesian_pos.transform.rotation.y,
            data.target_cartesian_pos.transform.rotation.z
        ])
        l_xyz = onp.array([
            data.target_cartesian_pos.transform.translation.x,
            data.target_cartesian_pos.transform.translation.y,
            data.target_cartesian_pos.transform.translation.z
        ])
        
        super().update_target_pose(
            side='left',
            position=l_xyz,
            wxyz=l_wxyz,
            gripper_state=data.target_gripper_pos,
            enable=data.enable
        )
        
    def _control_r(self, data):
        """Handle right controller updates."""
        r_wxyz = onp.array([
            data.target_cartesian_pos.transform.rotation.w,
            data.target_cartesian_pos.transform.rotation.x,
            data.target_cartesian_pos.transform.rotation.y,
            data.target_cartesian_pos.transform.rotation.z
        ])
        r_xyz = onp.array([
            data.target_cartesian_pos.transform.translation.x,
            data.target_cartesian_pos.transform.translation.y,
            data.target_cartesian_pos.transform.translation.z
        ])
        
        super().update_target_pose(
            side='right',
            position=r_xyz,
            wxyz=r_wxyz,
            gripper_state=data.target_gripper_pos,
            enable=data.enable
        )
        
    def start_episode(self):
        """Reset the environment and start a new episode."""
        self.image_primary.clear()
        self.image_wrist.clear()
        self.prev_action.clear()
        self.proprio_buffer.clear()
        self.action_queue.clear()
        self.cur_proprio = None
        self.home() # Move to home position as first action
        rospy.sleep(5)
        
def main(): 
    yumi_interface = YuMiDiffusionPolicyController()
    yumi_interface.run()
    
if __name__ == "__main__":
    tyro.cli(main)