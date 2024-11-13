"""01_kinematics.py
Tests robot forward + inverse kinematics using JaxMP.
"""

from typing import Literal, Optional
from pathlib import Path
import time
from loguru import logger
import tyro
import viser
import viser.extras

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

from jaxmp import JaxKinTree, RobotFactors
from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.extras.solve_ik import solve_ik

try:
    import sksparse
except ImportError:
    logger.info("sksparse not found. Some solvers may not work.")
    sksparse = None

import rospy
from sensor_msgs.msg import JointState # sensor message type
from abb_robot_msgs.srv import GetIOSignal, TriggerWithResultCode
from abb_egm_msgs.msg import EGMState
from std_msgs.msg import Float64, Float64MultiArray
from controller_manager_msgs.srv import SwitchController

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
    
def main(
    robot_description: Optional[str] = "yumi",
    robot_urdf_path: Optional[Path] = None,
):
    """
    Visualize the robot current config urdf.
    """
    # Load robot description.
    urdf = load_urdf(robot_description, robot_urdf_path)

    server = viser.ViserServer()

    # Visualize real robot via connection to ROS.
    urdf_vis_real = viser.extras.ViserUrdf(server, urdf, root_node_name="/base_real")
    urdf_vis_real.update_cfg(YUMI_REST_POSE)
    
    YUMI_CURR_POSE = {}
    def rws_ja_callback(data):
        get_io_signal = rospy.ServiceProxy('yumi/rws/get_io_signal', GetIOSignal)
        
        gripper_L = get_io_signal("hand_ActualPosition_L")
        gripper_R = get_io_signal("hand_ActualPosition_R")
        
        YUMI_CURR_POSE = {
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
            "gripper_r_joint": int(gripper_R.value)/10000,
            "gripper_l_joint": int(gripper_L.value)/10000,
        }
        
        urdf_vis_real.update_cfg(YUMI_CURR_POSE)
    

    rospy.Subscriber("yumi/rws/joint_states", JointState, rws_ja_callback)


if __name__ == "__main__":
    tyro.cli(main)