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
    
    def __init__(self, collect_data: bool = False, *args, **kwargs):
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
            rate.sleep() # Wait for first inputs to arrive
        
        assert type(self.observation_history) is torch.Tensor
        
        while not rospy.is_shutdown():
            
            input = {
            "observation": self.observation_history,
            "proprio": self.action
                }
        
            action_prediction = self.model.forward(input) # Denoise action prediction from obs and proprio...
            
            import pdb; pdb.set_trace() # TODO: Convert action_prediction to commands for YuMi
            
            super().solve_ik()
            super().update_visualization()
            
            # Publish joint commands
            # Need to flip arm order for actual robot control
            if self._first_js_callback:
                continue

            if self._homing:
                self.home()

            joint_desired = onp.array([
                self.joints[7], self.joints[8], self.joints[9],    # Left arm first
                self.joints[10], self.joints[11], self.joints[12], self.joints[13],
                self.joints[0], self.joints[1], self.joints[2],    # Right arm second
                self.joints[3], self.joints[4], self.joints[5], self.joints[6]
            ], dtype=onp.float32)
            
            # Publish joint commands
            msg = Float64MultiArray(data=joint_desired[0:14])
            self.joint_pub.publish(msg)
                
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
        
        # if not self._saving_data:
        #     self.handle_data(data)
        
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
        
    def handle_data(self, data):
        if self.collect_data:
            if data.traj_success and not self._saving_data:
                if not self.begin_record:
                    self.begin_record = True
                    self._homing = True
                    rospy.sleep(1.5)
                    self._homing = False
                    self.start_record()
                    rospy.sleep(0.5)
                    return None
                
                self._saving_data = True
                self.save_success()
                self._homing = True
                rospy.sleep(1.5)
                self._homing = False
                self.start_record()
                rospy.sleep(0.5)
                self._saving_data = False
                
            if data.traj_failure and not self._saving_data:
                if not self.begin_record:
                    self.begin_record = True
                    self._homing = True
                    rospy.sleep(1.5)
                    self._homing = False
                    self.start_record()
                    rospy.sleep(0.5)
                    return None
                
                self._saving_data = True
                self.save_failure()
                self._homing = True
                rospy.sleep(1.5)
                self._homing = False
                self.start_record()
                rospy.sleep(0.5)
                self._saving_data = False
    
    def _setup_collectors(self):
        if self.collect_data:
            self.start_record = rospy.ServiceProxy("/yumi_controller/start_recording", Empty)
            self.save_success = rospy.ServiceProxy("/yumi_controller/save_success", Empty)
            self.save_failure = rospy.ServiceProxy("/yumi_controller/save_failure", Empty)

def main(
    collect_data : bool = False
    ): 
    
    yumi_interface = YuMiDiffusionPolicyController(collect_data=collect_data)
        
    if collect_data:
        logger.info("Start data collection service")
        data_collector = DataCollector(init_node=False)
        yumi_interface._setup_collectors()
    yumi_interface.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)