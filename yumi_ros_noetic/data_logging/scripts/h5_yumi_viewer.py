
import tyro
import viser
import viser.extras
import numpy as np
import h5py
from pathlib import Path
from yumi_ros_noetic.base import YuMiBaseInterface
import time 
import plotly.express as px

def names_angles_to_dict(names, angles, idx=0):
    config = {
        names[0].decode("utf-8"): angles[idx, 0],
        names[1].decode("utf-8"): angles[idx, 1],
        names[2].decode("utf-8"): angles[idx, 2],
        names[3].decode("utf-8"): angles[idx, 3],
        names[4].decode("utf-8"): angles[idx, 4],
        names[5].decode("utf-8"): angles[idx, 5],
        names[6].decode("utf-8"): angles[idx, 6],
        names[7].decode("utf-8"): angles[idx, 7],
        names[8].decode("utf-8"): angles[idx, 8],
        names[9].decode("utf-8"): angles[idx, 9],
        names[10].decode("utf-8"): angles[idx, 10],
        names[11].decode("utf-8"): angles[idx, 11],
        names[12].decode("utf-8"): angles[idx, 12],
        names[13].decode("utf-8"): angles[idx, 13],
        names[14].decode("utf-8"): angles[idx, 14],
        names[15].decode("utf-8"): angles[idx, 15],
    }
    return config

def np_to_plotly(im):
    fig = px.imshow(im)
    
    fig.update_layout(
    coloraxis_showscale=False, 
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    margin=dict(l=0, r=0, t=0, b=0)  
    )
    return fig

def main(
    h5_file_path: str = '/home/xi/yumi_ros_noetic/trajectories/data/failure/2024/11/15/robot_trajectory_22_10_44.h5',
    ):
    
    h5_file_path = Path(h5_file_path)
    if not h5_file_path.exists():
        raise FileNotFoundError(f"{h5_file_path} does not exist.")
    
    yumi = YuMiBaseInterface(minimal=True)
    
    f = h5py.File(h5_file_path, 'r')
    
    with yumi.server.gui.add_folder("Controls"):
        next_button = yumi.server.gui.add_button("Next")
        prev_button = yumi.server.gui.add_button("Previous")
    
    slider_handle = yumi.server.gui.add_slider(
        "Data Entry Index", min=0, max=f['action/joint/joint_angle_rad'].shape[0]-1, step=1, initial_value=0
    )
    im = f['observation/camera/image/camera_rgb'][:]
    fig = np_to_plotly(im[0])
    
    fig_handle = yumi.server.gui.add_plotly(figure=fig, aspect=im[0].shape[0]/im[0].shape[1])
    names = f['action/joint/joint_name'][:].tolist()[0]
        
    angles = f['action/joint/joint_angle_rad'][:]
    config = names_angles_to_dict(names, angles, 0)
    
    cartesian = f['action/cartesian/cartesian_pos'][:]
    
    yumi.urdf_vis.update_cfg(config)
    
    tf_left_frame = yumi.server.scene.add_frame(
                    "tf_left",
                    axes_length=0.5 * 0.2,
                    axes_radius=0.01 * 0.2,
                    origin_radius=0.1 * 0.2,
                )
    tf_right_frame = yumi.server.scene.add_frame(
                    "tf_right",
                    axes_length=0.5 * 0.2,
                    axes_radius=0.01 * 0.2,
                    origin_radius=0.1 * 0.2,
                )
    
    tf_left_frame.position = cartesian[0, 4:7]
    tf_left_frame.wxyz = cartesian[0, 0:4]
    tf_right_frame.position = cartesian[0, 11:14]
    tf_right_frame.wxyz = cartesian[0, 7:11]
    
    @next_button.on_click
    def _(_) -> None:
        slider_handle.value += 1
    
    @prev_button.on_click
    def _(_) -> None:
        slider_handle.value -= 1
    
    @slider_handle.on_update
    def _(_) -> None:
        config = names_angles_to_dict(names, angles, slider_handle.value)
        
        tf_left_frame.position = cartesian[slider_handle.value, 4:7]
        tf_left_frame.wxyz = cartesian[slider_handle.value, 0:4]
        tf_right_frame.position = cartesian[slider_handle.value, 11:14]
        tf_right_frame.wxyz = cartesian[slider_handle.value, 7:11]
        
        yumi.urdf_vis.update_cfg(config)
        
        fig = np_to_plotly(im[slider_handle.value])
        fig_handle.figure = fig
        
        print("\nJoint Time:", f['action/joint/timestamp'][slider_handle.value][0]/1e9)
        print("Image Time:", f['observation/camera/image/timestamp'][slider_handle.value][0]/1e9)
        print("Difference:", f['observation/camera/image/timestamp'][slider_handle.value][0]/1e9 - f['action/joint/timestamp'][slider_handle.value][0]/1e9)
        
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)
