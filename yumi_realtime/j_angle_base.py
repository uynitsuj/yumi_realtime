from typing import Literal, Optional
import time
from dataclasses import dataclass
from loguru import logger

import numpy as onp
import viser
import viser.extras
from jaxmp.extras.urdf_loader import load_urdf
import jaxlie

import os

from pathlib import Path

@dataclass
class TransformHandle:
    """Data class to store transform handles."""
    frame: viser.FrameHandle

class YuMiJointAngleBaseInterface:
    """
    Base interface for YuMi robot visualization.
    - This class does not require ROS or real robot as this is a VIRTUAL representation, but serves as base class for the ROS interface
    - Running this file allows you to control a virtual YuMi robot in viser with transform handle gizmos.
    """
    
    YUMI_REST_POSE = {
        "yumi_joint_1_r": 1.21442839,
        "yumi_joint_2_r": -1.03205606,
        "yumi_joint_7_r": -1.10072738,
        "yumi_joint_3_r": 0.2987352 - 0.5,
        "yumi_joint_4_r": -1.85257716,
        "yumi_joint_5_r": 1.25363652,
        "yumi_joint_6_r": -2.42181893,
        "yumi_joint_1_l": -1.24839656,
        "yumi_joint_2_l": -1.09802876,
        "yumi_joint_7_l": 1.06634394,
        "yumi_joint_3_l": 0.31386161 - 0.5,
        "yumi_joint_4_l": 1.90125141,
        "yumi_joint_5_l": 1.3205139,
        "yumi_joint_6_l": 2.43563939,
        "gripper_r_joint": 0.025,
        "gripper_l_joint": 0.025,
    }

    def __init__(
        self,
        minimal: bool = False,
        slider_control: bool = True,
    ):
        self.minimal = minimal
        self.slider_control = slider_control
        # Set device

        # Initialize viser server
        self.server = viser.ViserServer()
        
        # Load robot description
        self.urdf = load_urdf(None, Path(os.path.dirname(os.path.abspath(__file__)) + "/../data/yumi_description/urdf/yumi.urdf"))
        self.rest_pose = onp.array(list(self.YUMI_REST_POSE.values()))

        # Target transform handle names HARDCODED BAD
        # self.target_names = ["left_dummy_joint", "right_dummy_joint"]
        
        self.joints = self.rest_pose
        
        # Setup visualization
        self._setup_visualization()
        
        if not minimal:
            self._setup_gui()
            if self.slider_control:
                self.create_robot_control_sliders()
        # Initialize state
        self.base_pose = jaxlie.SE3.identity()
        self.base_frame.position = onp.array(self.base_pose.translation())
        self.base_frame.wxyz = onp.array(self.base_pose.rotation().wxyz)
        
    def _setup_visualization(self):
        """Setup basic visualization elements."""
        # Add base frame and robot URDF
        self.base_frame = self.server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = viser.extras.ViserUrdf(
            self.server, 
            self.urdf, 
            root_node_name="/base"
        )
        self.urdf_vis.update_cfg(self.YUMI_REST_POSE)
        
        # Add ground grid
        self.server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)
        
    def _setup_gui(self):
        """Setup GUI elements."""
        # Add timing display
        self.timing_handle = self.server.gui.add_number("Time (ms)", 0.01, disabled=True)
        self.tf_size_handle = 0.2
    
    def update_visualization(self):
        """Update visualization with current state."""
        # Update base frame
        self.base_frame.position = onp.array(self.base_pose.translation())
        self.base_frame.wxyz = onp.array(self.base_pose.rotation().wxyz)
        
        # Update robot configuration
        self.urdf_vis.update_cfg(onp.array(self.joints))


    def create_robot_control_sliders(self) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
        """Create slider for each joint of the robot. We also update robot model
        when slider moves."""
        slider_handles: list[viser.GuiInputHandle[float]] = []
        initial_config: list[float] = []
        for joint_name, (
            lower,
            upper,
        ) in self.urdf_vis.get_actuated_joint_limits().items():
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi
            initial_pos = self.YUMI_REST_POSE[joint_name]
            slider = self.server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=0.5e-3,
                initial_value=initial_pos,
            )
            
            @slider.on_update
            def _(_):
                self.joints = onp.array([slider.value for slider in slider_handles])
                
            slider_handles.append(slider)
            initial_config.append(initial_pos)
        return slider_handles, initial_config


    def home(self):
        self.joints = self.rest_pose
        
    def run(self):
        """Main run loop."""
        while True:
            self.update_visualization()            

if __name__ == "__main__":
    yumi_interface = YuMiJointAngleBaseInterface()
    yumi_interface.run()