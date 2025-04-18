from yumi_realtime.j_angle_base import YuMiJointAngleBaseInterface
from loguru import logger
import viser
import tyro
# import jax.numpy as onp
import jaxlie
import numpy as onp
import time
import threading
from typing import Literal

import rospy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
from std_msgs.msg import Float64MultiArray, Header, String, Float64
from abb_robot_msgs.srv import GetIOSignal, SetIOSignal, TriggerWithResultCode
from controller_manager_msgs.srv import SwitchController
from abb_egm_msgs.msg import EGMState
from std_srvs.srv import Empty, EmptyResponse
from yumi_realtime.data_logging.data_collector import DataCollector

class YuMiJointAngleROSInterface(YuMiJointAngleBaseInterface):
    """YuMi joint angle control interface with ROS integration."""
    
    def __init__(self, collect_data: bool = False, *args, **kwargs):
        # Initialize base class first
        super().__init__(*args, **kwargs)
        
        self.ros_initialized = False
        self._first_js_callback = True
        self._interactive_handles = True
        self._js_update_lock = threading.Lock()
        self.collect_data = collect_data
        try:
            rospy.init_node('yumi_ja_controller')
            
            visible_topics = rospy.get_published_topics()
            
            if not any([topic[0] == '/yumi/rws/joint_states' for topic in visible_topics]):
                raise Exception("Searched for joint feedback topic '/yumi/rws/joint_states' and found nothing. Is the real YuMi ROS control interface running?")
            
            # Setup publishers
            self.joint_pub = rospy.Publisher(
                "yumi/egm/joint_group_position_controller/command", 
                Float64MultiArray, 
                queue_size=10
            )

            self.ja_combined_pub = rospy.Publisher(
                "/yumi/combined/joint_states",
                JointState,
                queue_size=10              
            )
            
            # Setup services
            self.get_io = rospy.ServiceProxy('yumi/rws/get_io_signal', GetIOSignal)
            self.set_io = rospy.ServiceProxy('yumi/rws/set_io_signal', SetIOSignal)
            self.start_egm = rospy.ServiceProxy(
                '/yumi/rws/sm_addin/start_egm_joint', 
                TriggerWithResultCode
            )
            self.switch_controller = rospy.ServiceProxy(
                '/yumi/egm/controller_manager/switch_controller', 
                SwitchController
            )
            
            self._setup_real_visualization()
            
            # Setup subscribers
            rospy.Subscriber(
                "yumi/rws/joint_states", 
                JointState, 
                self._joint_state_callback,
                queue_size=1
            )
            rospy.Subscriber(
                "yumi/egm/joint_states", 
                JointState, 
                self._joint_state_callback,
                queue_size=1
            )
            self.egm_js_counter = 0
            rospy.Subscriber(
                "yumi/egm/egm_states", 
                EGMState, 
                self._egm_state_callback
            )
            
            if self.collect_data:
                self._setup_collectors()
                self.add_gui_data_collection_controls()
            
            # Initialize gripper states
            self.prev_gripper_L = 0
            self.prev_gripper_R = 0
            self.gripper_L_pos = None
            self.gripper_R_pos = None
            self.egm_active = False
            self._homing = False

            # Start EGM
            self.start_egm_control()
            
            self.ros_initialized = True
            logger.info("ROS node initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ROS node: {e}")
            exit(1)
            
    def _setup_real_visualization(self):
        """Setup opqaue visualization of real robot state."""
        self.urdf_vis_real = viser.extras.ViserUrdf(
            self.server, 
            self.urdf, 
            root_node_name="/base_real",
            mesh_color_override=(0.65, 0.5, 0.5)
        )
        self.urdf_vis_real.update_cfg(self.YUMI_REST_POSE)
        
        # Make phantom robot slightly opqaue
        for mesh in self.urdf_vis_real._meshes:
            mesh.opacity = 0.4
            
    def start_egm_control(self):
        """Start EGM joint control."""
        try:
            rospy.wait_for_service('/yumi/rws/sm_addin/start_egm_joint')
            self.start_egm()
            rospy.sleep(0.1)
            
            self.switch_controller(
                start_controllers=['joint_group_position_controller'],
                stop_controllers=[''],
                strictness=3,
                start_asap=True,
                timeout=0.0
            )
        except Exception as e:
            logger.error(f"Failed to start EGM control: {e}")
    
    def _map_egm_joints(self, data: JointState):
        """Remap EGM joint state order."""
        joints_real = {
                    "yumi_joint_1_r": data.position[7],
                    "yumi_joint_2_r": data.position[8],
                    "yumi_joint_7_r": data.position[9],
                    "yumi_joint_3_r": data.position[10],
                    "yumi_joint_4_r": data.position[11],
                    "yumi_joint_5_r": data.position[12],
                    "yumi_joint_6_r": data.position[13],
                    "yumi_joint_1_l": data.position[0],
                    "yumi_joint_2_l": data.position[1],
                    "yumi_joint_7_l": data.position[2],
                    "yumi_joint_3_l": data.position[3],
                    "yumi_joint_4_l": data.position[4],
                    "yumi_joint_5_l": data.position[5],
                    "yumi_joint_6_l": data.position[6],
                    "gripper_r_joint": int(self.gripper_R_pos)/10000,
                    "gripper_l_joint": int(self.gripper_L_pos)/10000,
                }
        js_data_struct = JointState()
        js_data_struct.header = Header()
        js_data_struct.header.stamp = data.header.stamp
        js_data_struct.name = list(joints_real.keys())
        js_data_struct.position = list(joints_real.values())
        js_data_struct.velocity = list(data.velocity)[7:14] + list(data.velocity)[0:7]
       
        self.ja_combined_pub.publish(js_data_struct)
        
        return joints_real
    
    def _map_rws_joints(self, data: JointState):
        """Remap RWS joint state order."""
        joints_real = {
                    "yumi_joint_1_r": data.position[0],
                    "yumi_joint_2_r": data.position[1],
                    "yumi_joint_7_r": data.position[2],
                    "yumi_joint_3_r": data.position[3],
                    "yumi_joint_4_r": data.position[4],
                    "yumi_joint_5_r": data.position[5],
                    "yumi_joint_6_r": data.position[6],
                    "yumi_joint_1_l": data.position[7],
                    "yumi_joint_2_l": data.position[8],
                    "yumi_joint_7_l": data.position[9],
                    "yumi_joint_3_l": data.position[10],
                    "yumi_joint_4_l": data.position[11],
                    "yumi_joint_5_l": data.position[12],
                    "yumi_joint_6_l": data.position[13],
                    "gripper_r_joint": int(self.gripper_R_pos)/10000,
                    "gripper_l_joint": int(self.gripper_L_pos)/10000,
                }
        
        return joints_real
    
    def _joint_state_callback(self, data: JointState):
        """Handle joint state updates from the real robot."""
        try:
            with self._js_update_lock:
                # Get gripper states
                if len(data.velocity) == 0: # RWS joint states subscriber runs at ~5Hz
                    gripper_msg_L = self.get_io("hand_ActualPosition_L") 
                    gripper_msg_R = self.get_io("hand_ActualPosition_R")
                    self.gripper_L_pos = int(gripper_msg_L.value) if gripper_msg_L.value != '' else self.gripper_L_pos
                    self.gripper_R_pos = int(gripper_msg_R.value) if gripper_msg_R.value != '' else self.gripper_R_pos
                                    
                    # Update real robot joint configuration    
                    joints_real = self._map_rws_joints(data)
                else: # EGM joint states subscriber runs at 250Hz
                    if not self._first_js_callback:
                        joints_real = self._map_egm_joints(data)
                        self.egm_js_counter += 1
                        if self.egm_js_counter % 2 == 0:
                            self.egm_js_counter = 0
                            return 0
                    else:
                        gripper_msg_L = self.get_io("hand_ActualPosition_L") 
                        gripper_msg_R = self.get_io("hand_ActualPosition_R")
                        self.gripper_L_pos = int(gripper_msg_L.value) if gripper_msg_L.value != '' else self.gripper_L_pos
                        self.gripper_R_pos = int(gripper_msg_R.value) if gripper_msg_R.value != '' else self.gripper_R_pos
                        return 0
                assert type(self.gripper_L_pos) == int and type(self.gripper_R_pos) == int
                
                # Update real robot visualization     
                self.urdf_vis_real.update_cfg(joints_real)
                
                if self._first_js_callback:
                    logger.info(f"Received first joint state update for arm")
                    self.update_target_joints(
                        joints=self.joints,
                        enable=False
                    )
                    self._first_js_callback = False
                    
        except Exception as e:
            logger.error(f"Error in joint state callback: {e}")
        
    def home(self):
        self.joints = self.rest_pose
    
    def _egm_state_callback(self, data: EGMState):
        """Monitor EGM state and restart if necessary."""
        if data.egm_channels[0].active and data.egm_channels[1].active:
            self.egm_active = True
        else:
            self.egm_active = False
            self.start_egm_control()
            rospy.sleep(1.0)
            
    def update_target_joints(self, joints: onp.ndarray, enable: bool, thresh: float = 0.01):
        """
        Update target joint angles and gripper state
        joints should be in format
        "yumi_joint_1_r",
        "yumi_joint_2_r",
        "yumi_joint_7_r",
        "yumi_joint_3_r",
        "yumi_joint_4_r",
        "yumi_joint_5_r",
        "yumi_joint_6_r",
        "yumi_joint_1_l",
        "yumi_joint_2_l",
        "yumi_joint_7_l",
        "yumi_joint_3_l",
        "yumi_joint_4_l",
        "yumi_joint_5_l",
        "yumi_joint_6_l",
        "gripper_r_joint",
        "gripper_l_joint",
        """
        self.joints = joints
        self.joints[7:13] = list(self.YUMI_REST_POSE.values())[7:13]
        
        self.call_gripper("right", joints[-2] < thresh, enable)
        self.call_gripper("left", joints[-1] < thresh, enable)

    def call_gripper(self, side: Literal['left', 'right'], gripper_state: bool, enable: bool):       
        # Call gripper I/O services
        prev_gripper = self.prev_gripper_L if side == 'left' else self.prev_gripper_R
        if enable:
            if gripper_state:
                if prev_gripper != 4:
                    self.set_io(f"cmd_GripperState_{side[0].upper()}", "4")
                    time.sleep(0.05)
                    self.set_io("RUN_SG_ROUTINE", "1")
                    if side == 'left':
                        self.prev_gripper_L = 4
                    else:
                        self.prev_gripper_R = 4
                    time.sleep(0.05)
                    self.set_io("RUN_SG_ROUTINE", "0")
            else:
                if prev_gripper != 5:
                    self.set_io(f"cmd_GripperState_{side[0].upper()}", "5")
                    time.sleep(0.05)
                    self.set_io("RUN_SG_ROUTINE", "1")
                    if side == 'left':
                        self.prev_gripper_L = 5
                    else:
                        self.prev_gripper_R = 5
                    time.sleep(0.05)
                    self.set_io("RUN_SG_ROUTINE", "0")
    
    def call_gripper_pos(self, side: Literal['left', 'right'], gripper_state: float, enable: bool):
        # Call gripper position control I/O services
        """
        gripper_state: float in meters [0, 0.025]
        """
        prev_gripper = self.prev_gripper_L if side == 'left' else self.prev_gripper_R
        if enable:
            self.set_io(f"cmd_GripperState_{side[0].upper()}", "0")
            self.set_io(f"cmd_GripperState_{side[0].upper()}", "3")
            self.set_io(f"cmd_GripperPos_{side[0].upper()}", str(int(gripper_state*1000)))
            
            time.sleep(0.02)
            self.set_io("RUN_SG_ROUTINE", "1")
            if side == 'left':
                self.prev_gripper_L = 3
            else:
                self.prev_gripper_R = 3
            time.sleep(0.05)
            self.set_io("RUN_SG_ROUTINE", "0")
            time.sleep(0.05)

    def calib_gripper(self, side: Literal['left', 'right']):    
        # Call gripper calib I/O services
        prev_gripper = self.prev_gripper_L if side == 'left' else self.prev_gripper_R
        if prev_gripper != 2:
            self.set_io(f"cmd_GripperState_{side[0].upper()}", "2")
            time.sleep(0.05)
            self.set_io("RUN_SG_ROUTINE", "1")
            if side == 'left':
                self.prev_gripper_L = 2
            elif side == 'right':
                self.prev_gripper_R = 2
            else:
                raise ValueError(f"Invalid side {side}, must be 'left' or 'right'")
            time.sleep(0.05)
            self.set_io("RUN_SG_ROUTINE", "0")
    
    def add_gui_data_collection_controls(self):
        with self.server.gui.add_folder("Data Collection Controls"):
            self.start_record_button = self.server.gui.add_button("Start Recording")
            self.save_success_button = self.server.gui.add_button("Save Success", disabled=True)
            self.save_failure_button = self.server.gui.add_button("Save Failure", disabled=True)
            
        @self.start_record_button.on_click
        def _(_) -> None:
            """Callback for start recording."""
            self.start_record()
            self.start_record_button.disabled = True
            self.save_success_button.disabled = False
            self.save_failure_button.disabled = False
        @self.save_success_button.on_click
        def _(_) -> None:
            """Callback for save success."""
            self.save_success()
            self.start_record_button.disabled = False
            self.save_success_button.disabled = True
            self.save_failure_button.disabled = True
        @self.save_failure_button.on_click
        def _(_) -> None:
            """Callback for save failure."""
            self.save_failure()
            self.start_record_button.disabled = False
            self.save_success_button.disabled = True
            self.save_failure_button.disabled = True
        
    def add_gui_gripper_controls(self):
        while self.gripper_L_pos is None or self.gripper_R_pos is None:
            time.sleep(1.0)
        with self.server.gui.add_folder("Gripper Controls"):
            self.left_grip_gui_slider = self.server.gui.add_slider(
                "Left Gripper (mm)",
                min=0.0,
                max=25.0,
                step=1.5,
                initial_value=int(self.gripper_L_pos)/10,
            )
            self.joints[15] = int(self.gripper_L_pos)/10000
            self.bin_left_grip_button_group = self.server.gui.add_button_group(
                label="Left Gripper Actions",
                options=["Calibrate", "Open", "Close"],
            )
            self.right_grip_gui_slider = self.server.gui.add_slider(
                "Right Gripper (mm)",
                min=0.0,
                max=25.0,
                step=1.5,
                initial_value=int(self.gripper_R_pos)/10,
            )
            self.joints[14] = int(self.gripper_R_pos)/10000
            self.bin_right_grip_button_group = self.server.gui.add_button_group(
                label="Right Gripper Actions",
                options=["Calibrate", "Open", "Close"],
            )
        @self.left_grip_gui_slider.on_update
        def _(_) -> None:
            """Callback for left gripper control."""
            self.call_gripper_pos('left', self.left_grip_gui_slider.value/1000, True)
            self.joints[15] = self.left_grip_gui_slider.value/1000
        @self.right_grip_gui_slider.on_update
        def _(_) -> None:
            """Callback for right gripper control."""
            self.call_gripper_pos('right', self.right_grip_gui_slider.value/1000, True)
            self.joints[14] = self.right_grip_gui_slider.value/1000
        @self.bin_left_grip_button_group.on_click
        def _(_) -> None:
            """Callback for left gripper actions."""
            if self.bin_left_grip_button_group.value == "Calibrate":
                self.calib_gripper('left')
            elif self.bin_left_grip_button_group.value == "Open":
                self.left_grip_gui_slider.value = 25.0
            elif self.bin_left_grip_button_group.value == "Close":
                self.left_grip_gui_slider.value = 0.0
        @self.bin_right_grip_button_group.on_click
        def _(_) -> None:
            """Callback for right gripper actions."""
            if self.bin_right_grip_button_group.value == "Calibrate":
                self.calib_gripper('right')
            elif self.bin_right_grip_button_group.value == "Open":
                self.right_grip_gui_slider.value = 25.0
            elif self.bin_right_grip_button_group.value == "Close":
                self.right_grip_gui_slider.value = 0.0
    
    def run(self):
        """Override main run loop to include ROS control."""
        rate = rospy.Rate(150)  # 150Hz control loop          
        self.add_gui_gripper_controls()
        while not rospy.is_shutdown():
            # Run base class IK and visualization updates
            super().update_visualization()
                        
            if self._first_js_callback:
                continue

            if self._homing:
                self.home()

            self.publish_joint_commands()
    
            rate.sleep()
            
    def publish_joint_commands(self):
        """Publish joint commands to the robot."""
        # Need to flip arm order for actual robot control
        joint_desired = onp.array([
                self.joints[7], self.joints[8], self.joints[9],    # Left arm first
                self.joints[10], self.joints[11], self.joints[12], self.joints[13],
                self.joints[0], self.joints[1], self.joints[2],    # Right arm second
                self.joints[3], self.joints[4], self.joints[5], self.joints[6]
            ], dtype=onp.float32)
        msg = Float64MultiArray(data=joint_desired)
        self.joint_pub.publish(msg)
    
    def _setup_collectors(self):
        if self.collect_data:
            self.start_record = rospy.ServiceProxy("/yumi_controller/start_recording", Empty)
            self.save_success = rospy.ServiceProxy("/yumi_controller/save_success", Empty)
            self.save_failure = rospy.ServiceProxy("/yumi_controller/save_failure", Empty)
            self.stop_record = rospy.ServiceProxy("/yumi_controller/stop_recording", Empty)

def main(
    collect_data : bool = False,
    task_name : str = 'example_task1'
    ): 
    
    yumi_interface = YuMiJointAngleROSInterface(collect_data=collect_data)
    
    if collect_data:
        logger.info("Start data collection service")
        data_collector = DataCollector(init_node=False, task_name=task_name)

    yumi_interface.run()
    
    
if __name__ == "__main__":
    tyro.cli(main)