"""
References: 
https://github.com/droid-dataset/droid/blob/main/droid/controllers/oculus_controller.py
https://github.com/rail-berkeley/oculus_reader/blob/de73f3d259b3c41c4564f70a64682e24aa3ac31c/oculus_reader/visualize_oculus_transforms.py
"""

import time
import numpy as np
from oculus_reader.reader import OculusReader
from tf.transformations import quaternion_from_matrix, quaternion_matrix
import rospy
import tf2_ros
import threading
import geometry_msgs.msg
import std_msgs.msg
from vr_policy.msg import VRPolicyAction, OculusData
from scipy.spatial.transform import Rotation as R
from yumi_realtime.oculus_controller.utils.transformations import quat_to_euler, euler_to_quat, rmat_to_quat, quat_diff, add_angles, vec_to_reorder_mat, add_quats
import tyro
from typing import Literal
from scipy.spatial.transform import Rotation as R

def publish_transform(transform, name):
    translation = transform[:3, 3]

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'world'
    t.child_frame_id = name
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    quat = quaternion_from_matrix(transform)
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    br.sendTransform(t)

def parse_data(data : OculusData):
    """
    Parse the button data from the Oculus reader node.
    """
    left_pose = quaternion_matrix([data.left_controller_transform.transform.rotation.x,
                                    data.left_controller_transform.transform.rotation.y,
                                    data.left_controller_transform.transform.rotation.z,
                                    data.left_controller_transform.transform.rotation.w])
    left_pose[:3, 3] = np.array([data.left_controller_transform.transform.translation.x,
                        data.left_controller_transform.transform.translation.y,
                        data.left_controller_transform.transform.translation.z])
    right_pose = quaternion_matrix([data.right_controller_transform.transform.rotation.x,
                                        data.right_controller_transform.transform.rotation.y,
                                        data.right_controller_transform.transform.rotation.z,
                                        data.right_controller_transform.transform.rotation.w])
    right_pose[:3, 3] = np.array([data.right_controller_transform.transform.translation.x,
                    data.right_controller_transform.transform.translation.y,
                    data.right_controller_transform.transform.translation.z])
    pose = {
        "r" : right_pose,
        "l" : left_pose
    }
    button = {
        "A" : data.A, 
        "B" : data.B,
        "X" : data.X,
        "Y" : data.Y,
        "RThU" : data.RThU,
        "LThU" : data.LThU,
        "RJ" : data.RJ,
        "LJ" : data.LJ,
        "RG" : data.RG,
        "LG" : data.LG,
        "RTr" : data.RTr,
        "LTr" : data.LTr,
        "rightJS" : (data.right_joystick_x, data.right_joystick_y),
        "leftJS" : (data.left_joystick_x, data.left_joystick_y),
        "rightGrip" : (data.right_grip,),
        "leftGrip" : (data.left_grip,),
        "rightTrig" : (data.right_trigger,),
        "leftTrig" : (data.left_trigger,),
    }
    return pose, button

def action7d2tf(action : np.ndarray):
    assert action.shape == (7,)
    action_data = geometry_msgs.msg.TransformStamped()
    action_data.header = std_msgs.msg.Header()
    action_data.header.stamp = rospy.Time.now()
    action_data.transform.translation.x = action[0]
    action_data.transform.translation.y = action[1]
    action_data.transform.translation.z = action[2]
    action_data.transform.rotation.x = action[3]
    action_data.transform.rotation.y = action[4]
    action_data.transform.rotation.z = action[5]
    action_data.transform.rotation.w = action[6]
    return action_data

class VRPolicy:
    def __init__(
        self,
        right_controller: bool = True,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
        # rmat_reorder: list = [-2, -1, -3, 4], det = -1, inverts all rotation magnitudes
        rmat_reorder: list = [2, 1, -3, 4],
        init_node: bool = False
    ):
        # Initialize the ROS node
        if init_node:
            rospy.init_node('vr_policy_node', anonymous=True)
        
        self._state_lock = threading.Lock()
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self.reset_state()

        # Subscribe to the Oculus reader node for both controllers
        rospy.Subscriber('/oculus_reader/data', OculusData, self._oculus_data_callback)
        
        # Subscribe to the current robot pose
        if right_controller:
            rospy.Subscriber('/yumi/tf_left_real', geometry_msgs.msg.TransformStamped, self._robot_data_callback)
        else: 
            rospy.Subscriber('/yumi/tf_right_real', geometry_msgs.msg.TransformStamped, self._robot_data_callback)

        # publisher for actions
        if right_controller:
            self.action_publisher = rospy.Publisher('/vr_policy/control_l', VRPolicyAction, queue_size=10)
        else:
            self.action_publisher = rospy.Publisher('/vr_policy/control_r', VRPolicyAction, queue_size=10)

    def _oculus_data_callback(self, data):
        """
        Callback function to handle incoming data from the Oculus reader node.
        """
        # Update the internal state with the received data
        self._update_internal_state(data)
        
        # Generate an action based on the internal state
        action = self._calculate_action()
        
        if action is None: 
            return
        # Publish the generated action
        policy_action = VRPolicyAction()
        policy_action.target_cartesian_pos = action7d2tf(action["target_pose"])
        # import pdb; pdb.set_trace()
        # policy_action.target_gripper_pos = action["target_gripper_pos"]
        policy_action.target_gripper_pos = self._state["buttons"][self.controller_id.upper() + "Tr"]
        
        policy_action.target_cartesian_vel = action7d2tf(action["target_vel"])
        policy_action.target_gripper_vel = action["target_gripper_vel"]
        policy_action.enable = self._state["movement_enabled"]
        
        policy_action.traj_success = self._state["buttons"]["A"]
        policy_action.traj_failure = self._state["buttons"]["B"]
        
        self._publish_action(policy_action)

    def _robot_data_callback(self, data):
        """
        Callback function to handle incoming data from the robot pose subscriber.
        """
        # Update the internal state with the received data
        with self._state_lock:
            self._state["robot_pose"] = quaternion_matrix([data.transform.rotation.x,
                                                            data.transform.rotation.y,
                                                            data.transform.rotation.z,
                                                            data.transform.rotation.w])
            self._state["robot_pose"][:3, 3] = np.array([data.transform.translation.x,
                                                data.transform.translation.y,
                                                data.transform.translation.z])

    def _publish_action(self, action):
        """
        Publish the generated action.
        """
        self.action_publisher.publish(action)

    def run(self):
        """
        Run the ROS spin loop to keep the policy active.
        """
        rospy.spin()

    def reset_state(self):
        self._state = {
            "robot_pose" : None, 
            "poses": {},
            "buttons": {"A": False, "B": False, "X": False, "Y": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None
        self.last_gripper_state = None

    def _update_internal_state(self, data):
        # Read Controller
        poses, buttons = parse_data(data)
        # Determine Control Pipeline #
        toggled = self._state["movement_enabled"] != buttons[self.controller_id.upper() + "G"]
        self.update_sensor = self.update_sensor or buttons[self.controller_id.upper() + "G"]
        self.reset_orientation = self.reset_orientation or buttons[self.controller_id.upper() + "J"]
        self.reset_origin = self.reset_origin or toggled

        # Save Info #
        self._state["poses"] = poses
        self._state["buttons"] = buttons
        self._state["movement_enabled"] = buttons[self.controller_id.upper() + "G"]
        self._state["controller_on"] = True

        # Update Definition Of "Forward" #
        stop_updating = self._state["buttons"][self.controller_id.upper() + "J"] or self._state["movement_enabled"]
        if self.reset_orientation:
            rot_mat = np.asarray(self._state["poses"][self.controller_id])
            if stop_updating:
                self.reset_orientation = False
            # try to invert the rotation matrix, if not possible, then just use the identity matrix
            try:
                rot_mat = np.linalg.inv(rot_mat)
            except:
                print(f"exception for rot mat: {rot_mat}")
                rot_mat = np.eye(4)
                self.reset_orientation = True
            self.vr_to_global_mat = rot_mat

    def _process_reading(self):
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        vr_gripper = self._state["buttons"]["rightTrig" if self.controller_id == "r" else "leftTrig"][0]
        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(self):
        # Read Sensor #
        
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False
        
        # Read Observation
        with self._state_lock: 
            if self._state["robot_pose"] is None:
                return None
            robot_pos = self._state["robot_pose"][:3, 3]
            robot_rmat = self._state["robot_pose"][:3, :3]
        robot_quat = rmat_to_quat(robot_rmat)

        if not self._state["movement_enabled"]:
            zero_vel = np.zeros(7)
            zero_vel[-1] = 1 # xyzw, 0001
            if self.last_gripper_state is None:
                gripper_state = self.vr_state["gripper"]
            else:
                gripper_state = self.last_gripper_state
            return {
                "target_pose" : np.concatenate([robot_pos, robot_quat]),
                "target_gripper_pos" : gripper_state,
                "target_vel" : zero_vel,
                "target_gripper_vel" : 0
            }
        
        # Reset Origin On Release #
        if self.reset_origin:
            self.robot_origin = {"pos": robot_pos, "quat": robot_quat}
            self.vr_origin = {"pos": self.vr_state["pos"], "quat": self.vr_state["quat"]}
            self.reset_origin = False

        # Calculate Positional Action #
        robot_pos_offset = robot_pos - self.robot_origin["pos"]
        target_pos_offset = self.vr_state["pos"] - self.vr_origin["pos"]

        pos_action = target_pos_offset - robot_pos_offset

        robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])

        # world to world 
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action = quat_to_euler(quat_action)

        # Calculate Desired Pose #
        target_pos = pos_action + robot_pos

        target_quat = add_quats(quat_action, robot_quat)
        target_cartesian = np.concatenate([target_pos, target_quat])
        target_gripper = self.vr_state["gripper"]
        self.last_gripper_state = target_gripper

        # Scale Appropriately #
        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        gripper_action = 0 # TODO fix this! 
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)
        rot_vel = euler_to_quat(rot_vel)
        action = np.concatenate([np.clip(lin_vel, -1, 1), rot_vel])
        return {
            "target_pose" : target_cartesian, 
            "target_gripper_pos" : target_gripper, 
            "target_vel" : action, 
            "target_gripper_vel" : np.clip(gripper_vel, -1, 1)
        }

    def get_info(self):
        return {
            "success": self._state["buttons"]["A"] if self.controller_id == 'r' else self._state["buttons"]["X"],
            "failure": self._state["buttons"]["B"] if self.controller_id == 'r' else self._state["buttons"]["Y"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }