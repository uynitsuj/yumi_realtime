
import tyro
import viser
import viser.extras
import numpy as np
import h5py
from pathlib import Path
from yumi_realtime.base import YuMiBaseInterface
import time 

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

def main(
    h5_file_path: str = '/home/xi/yumi_realtime/trajectories/data/success/transfer_tiger_241204/robot_trajectory_2024_12_04_18_04_46.h5',
    ):
    
    h5_file_path = Path(h5_file_path)
    if not h5_file_path.exists():
        raise FileNotFoundError(f"{h5_file_path} does not exist.")
    
    yumi = YuMiBaseInterface(minimal=True)
    
    f = h5py.File(h5_file_path, 'r')
    play=False
    
    with yumi.server.gui.add_folder("Controls"):
        play_button = yumi.server.gui.add_button(label = "Play", icon=viser.Icon.PLAYER_PLAY_FILLED)
        pause_button = yumi.server.gui.add_button(label = "Pause", icon=viser.Icon.PLAYER_PAUSE_FILLED, visible=False)
        next_button = yumi.server.gui.add_button(label = "Forward", icon=viser.Icon.ARROW_BIG_RIGHT_FILLED)
        prev_button = yumi.server.gui.add_button(label = "Back", icon=viser.Icon.ARROW_BIG_LEFT_FILLED)
        
    slider_handle = yumi.server.gui.add_slider(
        "Data Entry Index", min=0, max=f['state/joint/joint_angle_rad'].shape[0]-1, step=1, initial_value=0
    )
    
    # Observation
    image_cache = {}
    viser_img_handles = {}
    
    for camera_name in f['observation'].keys():
        image_cache[camera_name] = f[f'observation/{camera_name}/image/camera_rgb'][:]
        
    with yumi.server.gui.add_folder("Observation"):
        for camera_name in f['observation'].keys():
            viser_img_handles[camera_name] = yumi.server.gui.add_image(
                image = image_cache[camera_name][0],
                label = camera_name
            )
            
    # State
    names = f['state/joint/joint_name'][:].tolist()[0]    
    angles = f['state/joint/joint_angle_rad'][:]
    config = names_angles_to_dict(names, angles, 0)
    cartesian = f['state/cartesian/cartesian_pose'][:]

    # Action
    if 'action' in f.keys():
        cartesian_action = f['action/cartesian_pose'][:]
        
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
    
    tf_left_action_frame = yumi.server.scene.add_frame(
                    "tf_left_action",
                    axes_length=0.5 * 0.2,
                    axes_radius=0.01 * 0.2,
                    origin_radius=0.1 * 0.2,
                )
    tf_right_action_frame = yumi.server.scene.add_frame(
                    "tf_right_action",
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

        if 'action' in f.keys():
            tf_left_action_frame.position = cartesian_action[slider_handle.value, 4:7]
            tf_left_action_frame.wxyz = cartesian_action[slider_handle.value, 0:4]
            tf_right_action_frame.position = cartesian_action[slider_handle.value, 11:14]
            tf_right_action_frame.wxyz = cartesian_action[slider_handle.value, 7:11]

        yumi.urdf_vis.update_cfg(config)
        
        for camera_name in f['observation'].keys():
            viser_img_handles[camera_name].image = image_cache[camera_name][slider_handle.value]
        
    @play_button.on_click
    def _(_) -> None:
        nonlocal play 
        play = True
        play_button.visible = False
        pause_button.visible = True
    
    @pause_button.on_click
    def _(_) -> None:
        nonlocal play
        play = False
        play_button.visible = True
        pause_button.visible = False
        
        
    while True:
        if play:
            slider_handle.value = (slider_handle.value + 1) % f['state/joint/joint_angle_rad'].shape[0]
            
        time.sleep(1/15)

if __name__ == "__main__":
    tyro.cli(main)
