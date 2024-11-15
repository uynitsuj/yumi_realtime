from yumi_ros_noetic.controller import YuMiROSInterface
from loguru import logger
import numpy as onp
import tyro
from typing import Literal

import rospy
from vr_policy.msg import VRPolicyAction
from yumi_ros_noetic.oculus_control.utils.vr_control import VRPolicy

class YuMiOculusInterface(YuMiROSInterface):
    """YuMi interface with Oculus VR control."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interactive_handles = False
        
        # Setup Oculus control subscribers
        rospy.Subscriber(
            "/vr_policy/control_l",
            VRPolicyAction,
            self._control_l_callback,
            queue_size=1
        )
        rospy.Subscriber(
            "/vr_policy/control_r",
            VRPolicyAction,
            self._control_r_callback,
            queue_size=1
        )
        
        logger.info("VR control interface initialized")
        
    def _control_l_callback(self, data: 'VRPolicyAction'):
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
        
    def _control_r_callback(self, data: 'VRPolicyAction'):
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

def main(
    controller : Literal["r", "l", "rl"] = "rl", # left and right controller
    ): 
    
    yumi_interface = YuMiOculusInterface()
    
    if "r" in controller: 
        logger.info("Start right controller")
        right_policy = VRPolicy(right_controller=True)
    if "l" in controller:
        logger.info("Start left controller")
        left_policy = VRPolicy(right_controller=False)
        
    yumi_interface.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)