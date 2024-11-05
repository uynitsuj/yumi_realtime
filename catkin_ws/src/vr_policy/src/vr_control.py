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
import geometry_msgs.msg
from vr_policy.msg import VRPolicyAction, OculusData
from scipy.spatial.transform import Rotation as R
from transformations import quat_to_euler, euler_to_quat, rmat_to_quat, quat_diff, add_angles, vec_to_reorder_mat, add_quats

def parse_data(data : OculusData):
    """
    Parse the button data from the Oculus reader node.
    """
    left_pose = quaternion_matrix([data.left_controller_transform.transform.rotation.x,
                                    data.right_controller_transform.transform.rotation.y,
                                    data.right_controller_transform.transform.rotation.z,
                                    data.right_controller_transform.transform.rotation.w])
    left_pose[:3, 3] = np.array([data.right_controller_transform.transform.translation.x,
                        data.right_controller_transform.transform.translation.y,
                        data.right_controller_transform.transform.translation.z])
    right_pose = quaternion_matrix([data.left_controller_transform.transform.rotation.x,
                                        data.left_controller_transform.transform.rotation.y,
                                        data.left_controller_transform.transform.rotation.z,
                                        data.left_controller_transform.transform.rotation.w])
    right_pose[:3, 3] = np.array([data.left_controller_transform.transform.translation.x,
                    data.left_controller_transform.transform.translation.y,
                    data.left_controller_transform.transform.translation.z])
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
        rmat_reorder: list = [-2, -1, -3, 4],
    ):
        # Initialize the ROS node
        rospy.init_node('vr_policy_node', anonymous=True)
        
        # Subscribe to the Oculus reader node for both controllers
        rospy.Subscriber('/oculus_reader/data', OculusData, self._oculus_data_callback)
        
        # Subscribe to the current robot pose
        rospy.Subscriber('/robotpose', geometry_msgs.msg.TransformStamped, self._robot_data_callback)

        # publisher for actions
        if right_controller:
            self.action_publisher = rospy.Publisher('/vr_policy/control_r', VRPolicyAction, queue_size=10)
        else:
            self.action_publisher = rospy.Publisher('/vr_policy/control_l', VRPolicyAction, queue_size=10)
        
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

    def _oculus_data_callback(self, data):
        """
        Callback function to handle incoming data from the Oculus reader node.
        """
        # Update the internal state with the received data
        self._update_internal_state(data)
        
        # Generate an action based on the internal state
        action = self._calculate_action())
        
        # Publish the action
        self._publish_action(action)

    def _robot_data_callback(self, data):
        """
        Callback function to handle incoming data from the robot pose subscriber.
        """
        # Update the internal state with the received data
        # TODO this needs a lock! 
        self._state["robot_pose"] = quaternion_matrix([data.transform.rotation.x,
                                                     data.transform.rotation.y,
                                                     data.transform.rotation.z,
                                                     data.transform.rotation.w])
        self._state["robot_pose"] = [data.transform.translation.x,
                                          data.transform.translation.y,
                                          data.transform.translation.z]

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

    def _calculate_action(self, include_info=False):
        # Read Sensor #
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        # Read Observation
        robot_pos = self._state["robot_pose"][:3, 3]
        robot_quat = rmat_to_quat(self._state["robot_pose"][:3, :3])
        # robot_gripper = state_dict["gripper_position"]

        # Reset Origin On Release #
        if self.reset_origin:
            self.robot_origin = {"pos": robot_pos, "quat": robot_quat}
            self.vr_origin = {"pos": self.vr_state["pos"], "quat": self.vr_state["quat"]}
            self.reset_origin = False

        # Calculate Positional Action #
        robot_pos_offset = robot_pos - self.robot_origin["pos"]
        target_pos_offset = self.vr_state["pos"] - self.vr_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset

        # Calculate Euler Action #
        robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action = quat_to_euler(quat_action)

        # Calculate Gripper Action #
        # gripper_action = (self.vr_state["gripper"] * 1.5) - robot_gripper

        # Calculate Desired Pose #
        target_pos = pos_action + robot_pos
        target_quat = add_quats(robot_quat, quat_action)
        # target_euler = add_angles(euler_action, robot_euler)
        target_cartesian = np.concatenate([target_pos, target_quat])
        target_gripper = self.vr_state["gripper"]

        # Scale Appropriately #
        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        # gripper_action *= self.gripper_action_gain
        gripper_action = 0 # TODO fix this! 
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)

        # Prepare Return Values #
        info_dict = {"target_cartesian_position": target_cartesian, "target_gripper_position": target_gripper}
        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        action = action.clip(-1, 1)

        # Return #
        if include_info:
            return action, info_dict
        else:
            return action

    def get_info(self):
        return {
            "success": self._state["buttons"]["A"] if self.controller_id == 'r' else self._state["buttons"]["X"],
            "failure": self._state["buttons"]["B"] if self.controller_id == 'r' else self._state["buttons"]["Y"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def forward(self, obs_dict, include_info=False):
        if self._state["poses"] == {}:
            action = np.zeros(7)
            if include_info:
                return action, {}
            else:
                return action
        return self._calculate_action(obs_dict["robot_state"], include_info=include_info)
