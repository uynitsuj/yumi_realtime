from yumi_realtime.controller import YuMiROSInterface
from loguru import logger
import numpy as onp
import tyro
from typing import Literal
import jaxlie

import rospy
from vr_policy.msg import VRPolicyAction
from yumi_realtime.oculus_controller.utils.vr_control import VRPolicy
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
        self._saving_data = False
        self._homing = True
        self.joint_noise = 0.05
        self.noise_home_noise = None
                    
        logger.info("VR control interface initialized")
        
    def _control_l_callback(self, data: VRPolicyAction):
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
            gripper_state=bool(data.target_gripper_pos),
            enable=data.enable
        )
        
        if not self._saving_data:
            self.handle_data(data)
        
    def _control_r_callback(self, data: VRPolicyAction):
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
            gripper_state=bool(data.target_gripper_pos),
            enable=data.enable
        )
        
    def sample_home_pose(self):
        self.noise_home_noise = onp.random.uniform(-self.joint_noise, self.joint_noise, size=self.rest_pose.shape)
        self.noise_home_noise = self.rest_pose + self.noise_home_noise

    def handle_data(self, data: VRPolicyAction):
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
                self.sample_home_pose()
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
                self.sample_home_pose()
                rospy.sleep(1.5)
                self._homing = False
                self.start_record()
                rospy.sleep(0.5)
                self._saving_data = False

    def home(self):
        if self.noise_home_noise is None:
            self.joints = self.rest_pose
        else:
            self.joints = self.noise_home_noise
        # Update real robot transform frames
        fk_frames = self.kin.forward_kinematics(self.joints.copy())
        
        for side, joint_name in [('left', 'yumi_joint_6_l'), ('right', 'yumi_joint_6_r')]:
            joint_idx = self.kin.joint_names.index(joint_name)
            T_target_world = self.base_pose @ jaxlie.SE3(fk_frames[joint_idx])
            self.update_target_pose(
                side=side,
                position=onp.array(T_target_world.translation()),
                wxyz=onp.array(T_target_world.rotation().wxyz),
                gripper_state=False,
                enable=False
            )

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
        data_collector = DataCollector(init_node=False, task_name='pickup_tiger_241226')
        yumi_interface._setup_collectors()
    yumi_interface.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)