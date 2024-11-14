from yumi_ros_noetic.yumi_ros_noetic.controller import YuMiROSInterface
from yumi_ros_noetic.base import TransformHandle
from loguru import logger
import viser
import jax.numpy as jnp
import jaxlie
import numpy as onp
import time

import rospy
from vr_policy.msg import VRPolicyAction

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

if __name__ == "__main__":
    yumi_interface = YuMiOculusInterface()
    yumi_interface.run()