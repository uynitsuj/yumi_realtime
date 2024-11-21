from yumi_realtime.controller import YuMiROSInterface
from loguru import logger
import numpy as onp
import tyro
from typing import Literal

import rospy
from vr_policy.msg import VRPolicyAction
from yumi_realtime.oculus_control.utils.vr_control import VRPolicy
from yumi_realtime.data_logging.data_collector import DataCollector
from std_srvs.srv import Empty, EmptyResponse

class YuMiOculusInterface(YuMiROSInterface):
    """YuMi interface with Oculus VR control."""
    
    def __init__(self, collect_data: bool = False, *args, **kwargs):
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
        
        self.collect_data = collect_data
        self.begin_record = False
        
        if self.collect_data:
            self._saving_data = False
            
            self.start_record = rospy.ServiceProxy("/data_collector/start_recording", Empty)()
            self.save_success = rospy.ServiceProxy("/data_collector/save_success", Empty)()
            self.save_failure = rospy.ServiceProxy("/data_collector/save_failure", Empty)()
                    
        logger.info("VR control interface initialized")
        
    def _control_l_callback(self, data):
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
        
        self.handle_data(data)
        
    def _control_r_callback(self, data):
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
        
        self.handle_data(data)

    def handle_data(self, data):
        if self.collect_data:
            if data.traj_success and not self._saving_data:
                if not self.begin_record:
                    self.start_record()
                    return None
                self._saving_data = True
                self.save_success()
                
            if data.traj_failure and not self._saving_data:
                if not self.begin_record:
                    self.start_record()
                    return None
                self._saving_data = True
                self.save_failure()

def main(
    controller : Literal["r", "l", "rl"] = "rl", # left and right controller
    collect_data : bool = True
    ): 
    
    yumi_interface = YuMiOculusInterface(collect_data=collect_data)
    
    if "r" in controller: 
        logger.info("Start right controller")
        right_policy = VRPolicy(right_controller=True)
    if "l" in controller:
        logger.info("Start left controller")
        left_policy = VRPolicy(right_controller=False)
        
    if collect_data:
        logger.info("Start data collection service")
        data_collector = DataCollector(init_node=False)
        
    yumi_interface.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)