from yumi_realtime.controller import YuMiROSInterface
from loguru import logger
import numpy as onp
import tyro
from typing import Literal

import rospy

from yumi_realtime.data_logging.data_collector import DataCollector
from dp_gs.policy.model import DiffusionPolicy
from dp_gs.policy.diffusion_wrapper import DiffusionPolicyWrapper, normalize, unnormalize
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Header, String, Float64
from std_srvs.srv import Empty, EmptyResponse


import torch

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
        
        self.observation_history = []
        
        self.bridge = CvBridge()
        
        logger.info("Diffusion Policy controller initialized")
    
    def run(self):
        """Diffusion Policy controller loop."""
        rate = rospy.Rate(250) # 250Hz control loop          
        
        while ((self.height is None or self.width is None) or (self.cartesian_pose_L is None or self.cartesian_pose_R is None)):
            self.home() # Move to home position as first action
            rospy.sleep(5)
            rate.sleep() # Wait for first inputs to arrive
        
        assert type(self.observation_history) is torch.Tensor
        
        while not rospy.is_shutdown():
            
            input = {
            "observation": self.observation_history,
            "proprio": self.state # TODO: Update with proprioception data format from self.cartesian_pose_L and self.cartesian_pose_R
                }
        
            action_prediction = self.model.forward(input) # Denoise action prediction from obs and proprio...
            
            import pdb; pdb.set_trace() # TODO: Convert denoised action_prediction format to commands for YuMi
            
            super().solve_ik()
            super().update_visualization()
            
            # Publish joint commands
            # Need to flip arm order for actual robot control
            if self._first_js_callback:
                continue

            if self._homing:
                self.home()

            self.publish_joint_commands()
                
            rate.sleep()
        
    def image_callback(self, image_msg):
        """Handle camera observation updates."""
        if self.height is None and self.width is None:
            self.height = image_msg.height
            self.width = image_msg.width
            logger.info(f"First image received; Observation dim: {self.height}x{self.width}x3")
            
            # Reinitialize observation history with correct dimensions
            self.observation_history = torch.zeros(
                (self.model.obs_horizon, self.height, self.width, 3), dtype=torch.float32
            )
            
        np_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8').astype("float32") / 255.0  # Normalized to float [0, 1]
        
        # Convert NumPy array to PyTorch tensor and add to history
        new_obs = torch.from_numpy(np_img).permute(2, 0, 1)  # Convert HWC to CHW format (PyTorch standard)
        self.observation_history = torch.cat(
            (self.observation_history[1:], new_obs.unsqueeze(0)), dim=0
        )  # Maintain fixed-size queue

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
        
        self.update_target_pose(
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
        
        self.update_target_pose(
            side='right',
            position=r_xyz,
            wxyz=r_wxyz,
            gripper_state=data.target_gripper_pos,
            enable=data.enable
        )
        
def main(): 
    
    yumi_interface = YuMiDiffusionPolicyController()
    
    yumi_interface.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)