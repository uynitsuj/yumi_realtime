"""Robot URDF visualizer

Requires yourdfpy and robot_descriptions. Any URDF supported by yourdfpy should work.
- https://github.com/robot-descriptions/robot_descriptions.py
- https://github.com/clemense/yourdfpy

The :class:`viser.extras.ViserUrdf` is a lightweight interface between yourdfpy
and viser. It can also take a path to a local URDF file as input.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf
from pathlib import Path
import trimesh
import os
import cv2

import matplotlib.pyplot as plt
from datetime import datetime

# ROS imports
import rospy
from sensor_msgs.msg import JointState # sensor message type
# import abb_robot_driver_interfaces.abb_robot_msgs as robot_msgs # interface message type

YUMI_REST_POSE = {
    "yumi_joint_1_r": 1.21442839,
    "yumi_joint_2_r": -1.03205606,
    "yumi_joint_7_r": -1.10072738,
    "yumi_joint_3_r": 0.2987352 - 0.2,
    "yumi_joint_4_r": -1.85257716,
    "yumi_joint_5_r": 1.25363652,
    "yumi_joint_6_r": -2.42181893,
    "yumi_joint_1_l": -1.24839656,
    "yumi_joint_2_l": -1.09802876,
    "yumi_joint_7_l": 1.06634394,
    "yumi_joint_3_l": 0.31386161 - 0.2,
    "yumi_joint_4_l": 1.90125141,
    "yumi_joint_5_l": 1.3205139,
    "yumi_joint_6_l": 2.43563939,
    "gripper_r_joint": 0, # 0.025,
    "gripper_l_joint": 0, # 0.025,
}

def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        # initial_pos = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
        initial_pos = YUMI_REST_POSE[joint_name]
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main() -> None:
    # Start viser server.
    server = viser.ViserServer()
    
    def callback(data):

        YUMI_POSE = {
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
            "gripper_r_joint": 0,
            "gripper_l_joint": 0,
        }
        viser_urdf.update_cfg(YUMI_POSE)
        print(YUMI_POSE)

    # ViserUrdf expects the path to be a path object or a yourdfpy.URDF object
    urdf_path = Path("/home/xi/yumi_ros_noetic/data/yumi_description/urdf/yumi.urdf")
    viser_urdf = ViserUrdf(
            server, urdf_path
        )
    # Set initial robot configuration.
    viser_urdf.update_cfg(YUMI_REST_POSE)

    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )
        
    # Wait 2 seconds for viser before beginnning ros subscriber (for egm safety, can decide if necessary)
    time.sleep(2) 
    
    # Main loop thread
    while True:
        
        rospy.init_node('yumi_control', anonymous=True)
        
        rospy.Subscriber("yumi/rws/joint_states", JointState, callback)
        rospy.spin()
        # for i in range(0, len(slider_handles)):
        #     print(slider_handles[i].value)
        




if __name__ == "__main__":
    tyro.cli(main)