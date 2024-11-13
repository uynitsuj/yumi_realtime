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
from abb_robot_msgs.srv import GetIOSignal, SetIOSignal, TriggerWithResultCode
from abb_egm_msgs.msg import EGMState
from std_msgs.msg import Float64, Float64MultiArray
from controller_manager_msgs.srv import SwitchController

from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from geometry_msgs.msg import Transform, Vector3, Quaternion
from vr_policy.msg import VRPolicyAction

from multiprocessing import shared_memory

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
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    device: Literal["cpu", "gpu"] = "cpu",
    robot_description: Optional[str] = "yumi",
    robot_urdf_path: Optional[Path] = None,
):
    """
    Test robot inverse kinematics using JaxMP.
    Args:
        pos_weight: Weight for position error in IK.
        rot_weight: Weight for rotation error in IK.
        rest_weight: Weight for rest pose in IK.
        limit_weight: Weight for joint limits in IK.
        device: Device to use.
        robot_description: Name of the robot description to load.
        robot_urdf_path: Path to the robot URDF file.
    """
    # Set device.
    jax.config.update("jax_platform_name", device)

    # Load robot description.
    urdf = load_urdf(robot_description, robot_urdf_path)

    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = jnp.array(list(YUMI_REST_POSE.values()))
    JointVar = RobotFactors.get_var_class(kin, rest_pose)

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")
    urdf_vis.update_cfg(YUMI_REST_POSE)
    
    # Visualize real robot via connection to ROS.
    urdf_vis_real = viser.extras.ViserUrdf(server, urdf, root_node_name="/base_real", mesh_color_override=(0.65, 0.5, 0.5))
    urdf_vis_real.update_cfg(YUMI_REST_POSE)
    for mesh in urdf_vis_real._meshes:
        mesh.opacity = 0.4
        
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Add base-frame freezing logic.
    T_base_world_handles = []
    with server.gui.add_folder("T_base_world"):
        for dof in ["x", "y", "z", "rx", "ry", "rz"]:
            T_base_world_handles.append(
                server.gui.add_checkbox(f"Freeze {dof}", initial_value=True)
            )

    def get_freeze_base_xyz_xyz() -> jnp.ndarray:
        return jnp.array([handle.value for handle in T_base_world_handles]).astype(
            jnp.float32
        )

    # Add base-frame freezing logic.
    dof_target_handles = []
    with server.gui.add_folder("Target pose DoF"):
        for dof in ["x", "y", "z", "rx", "ry", "rz"]:
            dof_target_handles.append(
                server.gui.add_checkbox(f"Freeze {dof}", initial_value=True)
            )

    def get_freeze_target_xyz_xyz() -> jnp.ndarray:
        return jnp.array([handle.value for handle in dof_target_handles]).astype(
            jnp.float32
        )

    ConstrainedSE3Var = RobotFactors.get_constrained_se3(get_freeze_base_xyz_xyz())

    def update_constrained_se3_var():
        nonlocal ConstrainedSE3Var
        ConstrainedSE3Var = RobotFactors.get_constrained_se3(get_freeze_base_xyz_xyz())

    for handle in T_base_world_handles:
        handle.on_update(lambda _: update_constrained_se3_var())

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    tf_size_handle = server.gui.add_slider(
        "Gizmo size", min=0.01, max=0.4, step=0.01, initial_value=0.2
    )

    if sksparse is None:
        solver_types = ("conjugate_gradient", "dense_cholesky")
    else:
        solver_types = ("conjugate_gradient", "dense_cholesky", "cholmod")
    solver_type_handle = server.gui.add_dropdown(
        "Solver type",
        solver_types,
        initial_value="conjugate_gradient",
    )

    smooth_handle = server.gui.add_checkbox("Smooth", initial_value=True)

    with server.gui.add_folder("Manipulability"):
        manipulabiltiy_weight_handler = server.gui.add_slider(
            "weight", 0.0, 0.01, 0.001, 0.00
        )
        manipulability_cost_handler = server.gui.add_number(
            "Yoshikawa index", 0.001, disabled=True
        )
        

    target_frame_handles: list[viser.FrameHandle] = []

    # Put robot to rest pose :-)
    base_pose = jaxlie.SE3.identity()
    joints = rest_pose

    urdf_base_frame.position = onp.array(base_pose.translation())
    urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)
    urdf_vis.update_cfg(onp.array(joints))

    # Add relevant control frames to viz.
    # import pdb; pdb.set_trace()
    target_names = ['yumi_joint_6_l', 'yumi_joint_6_r']

    frame_handle_left_real = server.scene.add_frame(
        f"tf_left_real",
        axes_length=0.5 * tf_size_handle.value,
        axes_radius=0.01 * tf_size_handle.value,
        origin_radius=0.1 * tf_size_handle.value,
    )
    frame_handle_right_real = server.scene.add_frame(
        f"tf_right_real",
        axes_length=0.5 * tf_size_handle.value,
        axes_radius=0.01 * tf_size_handle.value,
        origin_radius=0.1 * tf_size_handle.value,
    )
    
    frame_handle_left_target = server.scene.add_frame(
        f"tf_left_target",
        axes_length=0.5 * tf_size_handle.value,
        axes_radius=0.01 * tf_size_handle.value,
        origin_radius=0.1 * tf_size_handle.value,
    )
    frame_handle_right_target = server.scene.add_frame(
        f"tf_right_target",
        axes_length=0.5 * tf_size_handle.value,
        axes_radius=0.01 * tf_size_handle.value,
        origin_radius=0.1 * tf_size_handle.value,
    )
    
    target_frame_handles = [frame_handle_left_target, frame_handle_right_target]
    
    print("Starting ROS Node")
    rospy.init_node('yumi_controller', anonymous=True)
    
    get_io_signal = rospy.ServiceProxy('yumi/rws/get_io_signal', GetIOSignal)
    set_io_signal = rospy.ServiceProxy('yumi/rws/set_io_signal', SetIOSignal)
    prev_gripper_L = 0
    prev_gripper_R = 0
    def control_l_callback(data):
        nonlocal prev_gripper_L
        l_wxyz = onp.array(
            [data.target_cartesian_pos.transform.rotation.w,
            data.target_cartesian_pos.transform.rotation.x,
            data.target_cartesian_pos.transform.rotation.y,
            data.target_cartesian_pos.transform.rotation.z]
        )
        l_xyz = onp.array(
            [data.target_cartesian_pos.transform.translation.x,
            data.target_cartesian_pos.transform.translation.y,
            data.target_cartesian_pos.transform.translation.z]
        )
        if data.enable:
            frame_handle_left_target.position = l_xyz
            frame_handle_left_target.wxyz = l_wxyz
        else:
            frame_handle_left_target.position = frame_handle_left_real.position
            frame_handle_left_target.wxyz = frame_handle_left_real.wxyz
        
        l_gripper_target = data.target_gripper_pos # 0.0 to 1.0
        if bool(l_gripper_target):
            if prev_gripper_L != 4 and data.enable:
                set_io_signal("cmd_GripperState_L", "4")
                time.sleep(0.05)
                set_io_signal("RUN_SG_ROUTINE", "1")
                prev_gripper_L = 4
                time.sleep(0.15)
                set_io_signal("RUN_SG_ROUTINE", "0")
        else:
            if prev_gripper_L != 5 and data.enable:
                set_io_signal("cmd_GripperState_L", "5")
                time.sleep(0.05)
                set_io_signal("RUN_SG_ROUTINE", "1")
                prev_gripper_L = 5
                time.sleep(0.15)
                set_io_signal("RUN_SG_ROUTINE", "0")

            
    def control_r_callback(data):
        nonlocal prev_gripper_R
        r_wxyz = onp.array(
            [data.target_cartesian_pos.transform.rotation.w,
            data.target_cartesian_pos.transform.rotation.x,
            data.target_cartesian_pos.transform.rotation.y,
            data.target_cartesian_pos.transform.rotation.z]
        )
        r_xyz = onp.array(
            [data.target_cartesian_pos.transform.translation.x,
            data.target_cartesian_pos.transform.translation.y,
            data.target_cartesian_pos.transform.translation.z]
        )
        if data.enable:
            frame_handle_right_target.position = r_xyz
            frame_handle_right_target.wxyz = r_wxyz
        else:
            frame_handle_right_target.position = frame_handle_right_real.position
            frame_handle_right_target.wxyz = frame_handle_right_real.wxyz
        
        r_gripper_target = data.target_gripper_pos # 0.0 to 1.0
        if bool(r_gripper_target):
            if prev_gripper_R != 4 and data.enable:
                set_io_signal("cmd_GripperState_R", "4")
                time.sleep(0.1)
                set_io_signal("RUN_SG_ROUTINE", "1")
                prev_gripper_R = 4
                time.sleep(0.15)
                set_io_signal("RUN_SG_ROUTINE", "0")
        else:
            if prev_gripper_R != 5 and data.enable:
                set_io_signal("cmd_GripperState_R", "5")
                time.sleep(0.1)
                set_io_signal("RUN_SG_ROUTINE", "1")
                prev_gripper_R = 5
                time.sleep(0.15)
                set_io_signal("RUN_SG_ROUTINE", "0")
            
                
    rospy.Subscriber("/vr_policy/control_l", VRPolicyAction, control_l_callback) # Controls left YuMi Arm, may be right-handed controller
    rospy.Subscriber("/vr_policy/control_r", VRPolicyAction, control_r_callback)
    
    start_egm = rospy.ServiceProxy('/yumi/rws/sm_addin/start_egm_joint', TriggerWithResultCode)
    switch_controller = rospy.ServiceProxy('/yumi/egm/controller_manager/switch_controller', SwitchController)
    # Start EGM joint control.
    
    def start_egm_control():
        rospy.wait_for_service('/yumi/rws/sm_addin/start_egm_joint')
        se_result = start_egm()

        time.sleep(0.1)
        sc_result = switch_controller(['joint_group_position_controller'],[''],3,True,0.0)
        
    EGMActive = False
    def egm_state_callback(data):
        if data.egm_channels[0].active and data.egm_channels[1].active:
            EGMActive = True
            # print("EGM Active")
        else:
            EGMActive = False
            start_egm_control()
            time.sleep(1)
        
    rospy.Subscriber("yumi/egm/egm_states", EGMState, egm_state_callback)

        
    def _(_):
        nonlocal joints
        base_pose = jnp.array(
            urdf_base_frame.wxyz.tolist() + urdf_base_frame.position.tolist()
        )

        for target_frame_handle, target_name in zip(
            target_frame_handles, target_names
        ):
            target_joint_idx = kin.joint_names.index(target_name)
            T_target_world = jaxlie.SE3(base_pose) @ jaxlie.SE3(
                kin.forward_kinematics(joints)[target_joint_idx]
            )

            target_frame_handle.position = onp.array(T_target_world.translation())
            target_frame_handle.wxyz = onp.array(T_target_world.rotation().wxyz)
            
    def rws_ja_callback(data):
        
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
        
        joints_real = jnp.array(list(YUMI_CURR_POSE.values()), dtype=jnp.float32)
        
        urdf_vis_real.update_cfg(YUMI_CURR_POSE)
        base_pose = jnp.array(urdf_base_frame.wxyz.tolist() + urdf_base_frame.position.tolist())        
        tf_left_real = jaxlie.SE3(base_pose) @ jaxlie.SE3(
            kin.forward_kinematics(joints_real)[kin.joint_names.index('yumi_joint_6_l')]
        )
        tf_right_real = jaxlie.SE3(base_pose) @ jaxlie.SE3(
            kin.forward_kinematics(joints_real)[kin.joint_names.index('yumi_joint_6_r')]
        )
        frame_handle_left_real.position = onp.array(tf_left_real.translation())
        frame_handle_left_real.wxyz = onp.array(tf_left_real.rotation().wxyz)
        
        frame_handle_right_real.position = onp.array(tf_right_real.translation())
        frame_handle_right_real.wxyz = onp.array(tf_right_real.rotation().wxyz)
        
        l_transform_stamped = TransformStamped()
        r_transform_stamped = TransformStamped()
        
        l_transform_stamped.header = Header()
        l_transform_stamped.header.stamp = rospy.Time.now()
        r_transform_stamped.header = Header()
        r_transform_stamped.header.stamp = rospy.Time.now()

        l_transform_stamped.transform = Transform()
        r_transform_stamped.transform = Transform()
        l_transform_stamped.transform.translation = Vector3(tf_left_real.translation()[0], tf_left_real.translation()[1], tf_left_real.translation()[2])
        r_transform_stamped.transform.translation = Vector3(tf_right_real.translation()[0], tf_right_real.translation()[1], tf_right_real.translation()[2])
        
        l_transform_stamped.transform.rotation = Quaternion(tf_left_real.rotation().wxyz[1], tf_left_real.rotation().wxyz[2], tf_left_real.rotation().wxyz[3], tf_left_real.rotation().wxyz[0])
        r_transform_stamped.transform.rotation = Quaternion(tf_right_real.rotation().wxyz[1], tf_right_real.rotation().wxyz[2], tf_right_real.rotation().wxyz[3], tf_right_real.rotation().wxyz[0])
        
        tf_left_real_pub.publish(l_transform_stamped)
        tf_right_real_pub.publish(r_transform_stamped)
    
    has_jitted = False
    while True:

        target_joint_indices = jnp.array(
            [
                kin.joint_names.index(target_name)
                for target_name in target_names
            ]
        )
        target_pose_list = [
            jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
            for target_tf_handle in target_frame_handles
        ]
        target_poses = jaxlie.SE3(
            jnp.stack([pose.wxyz_xyz for pose in target_pose_list])
        )
        manipulability_weight = manipulabiltiy_weight_handler.value

        if smooth_handle.value:
            initial_pose = joints
            joint_vel_weight = limit_weight
        else:
            initial_pose = rest_pose
            joint_vel_weight = 0.0

        ik_weight = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
        ik_weight = ik_weight * get_freeze_target_xyz_xyz()
        manipulability_weight = manipulabiltiy_weight_handler.value

        # Solve!
        start_time = time.time()
        base_pose, joints = solve_ik(
            kin,
            target_poses,
            target_joint_indices,
            initial_pose,
            JointVar,
            ik_weight,
            ConstrainedSE3Var=ConstrainedSE3Var,
            rest_weight=rest_weight,
            limit_weight=limit_weight,
            joint_vel_weight=joint_vel_weight,
            use_manipulability=(manipulability_weight > 0),
            manipulability_weight=manipulability_weight,
            solver_type=solver_type_handle.value,
        )

        # Ensure all computations are complete before measuring time
        jax.block_until_ready((base_pose, joints))
        timing_handle.value = (time.time() - start_time) * 1000
        if not has_jitted:
            logger.info("JIT compile + running took {} ms.", timing_handle.value)
            has_jitted = True
            tf_left_real_pub = rospy.Publisher("yumi/tf_left_real", TransformStamped, queue_size=10)
            tf_right_real_pub = rospy.Publisher("yumi/tf_right_real", TransformStamped, queue_size=10)
            rospy.Subscriber("yumi/rws/joint_states", JointState, rws_ja_callback)
            joint_pub = rospy.Publisher("yumi/egm/joint_group_position_controller/command", Float64MultiArray, queue_size=10)
            r = rospy.Rate(50) # 100Hz

        # Update visualizations.
        urdf_base_frame.position = onp.array(base_pose.translation())
        urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)
        urdf_vis.update_cfg(onp.array(joints))
        for target_frame_handle, target_joint_idx in zip(
            target_frame_handles, target_joint_indices
        ):
            T_target_world = base_pose @ jaxlie.SE3(
                kin.forward_kinematics(joints)[target_joint_idx]
            )
            target_frame_handle.position = onp.array(T_target_world.translation())
            target_frame_handle.wxyz = onp.array(T_target_world.rotation().wxyz)

        # Update manipulability cost.
        manip_cost = 0
        for target_joint_idx in target_joint_indices:
            manip_cost += RobotFactors.manip_yoshikawa(kin, joints, target_joint_idx)
        manip_cost /= len(target_joint_indices)
        manipulability_cost_handler.value = onp.array(manip_cost).item()
        
        # print(EGMActive)
            
        # Need to flip arm order
        joint_desired = onp.array([
            joints[7], joints[8], joints[9], joints[10], joints[11], joints[12], joints[13],
            joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], joints[6]
        ], dtype=onp.float32)
        
        ja_msg = Float64MultiArray()
        ja_msg.data = joint_desired[0:14]
        joint_pub.publish(ja_msg)
        r.sleep()

if __name__ == "__main__":
    tyro.cli(main)