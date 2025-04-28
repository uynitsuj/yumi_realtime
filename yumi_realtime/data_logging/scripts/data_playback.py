import tyro
import viser
import viser.extras
import numpy as np
import h5py
import time
import os
import glob
import cv2
from pathlib import Path
from typing import Dict, List, Optional
from yumi_realtime.base import YuMiBaseInterface


class H5TrajectoryViewer:
    def __init__(self, trajectory_dir: str):
        self.trajectory_dir = Path(trajectory_dir)
        if not self.trajectory_dir.exists():
            raise FileNotFoundError(f"{self.trajectory_dir} does not exist.")
        
        # Find all H5 files in the directory
        self.trajectories = self._find_h5_files(self.trajectory_dir)
        if not self.trajectories:
            raise ValueError(f"No H5 trajectory files found in {self.trajectory_dir}")
        
        # Initialize the YuMi interface
        self.yumi = YuMiBaseInterface(minimal=True)
        self.server = self.yumi.server
        self.play = False
        
        # Set the current trajectory to the first one found
        self.current_trajectory = self.trajectories[0]
        
        # Load the H5 file and relevant data
        self._load_trajectory(self.current_trajectory)
        
        # Setup the visualization
        self._setup_viser_scene()
        self._setup_viser_gui()
    
    def _find_h5_files(self, directory: Path) -> List[Path]:
        """Find all H5 files in the given directory and subdirectories."""
        h5_files = []
        
        # Look for .h5 files directly in the directory
        h5_files.extend(list(directory.glob("*.h5")))
        
        # Look for .h5 files in subdirectories (single level)
        for subdir in directory.iterdir():
            if subdir.is_dir():
                h5_files.extend(list(subdir.glob("*.h5")))
        
        # Sort by name for consistent ordering
        return sorted(h5_files)
    
    def _load_trajectory(self, trajectory_path: Path):
        """Load data from the specified H5 file."""
        # Pause playback when loading a new trajectory
        was_playing = self.play
        if was_playing:
            self.play = False
            if hasattr(self, 'play_button') and hasattr(self, 'pause_button'):
                self.play_button.visible = True
                self.pause_button.visible = False
        
        # Close any previously opened file
        if hasattr(self, 'f') and self.f:
            self.f.close()
            
        self.current_trajectory = trajectory_path
        print(f"Loading trajectory: {trajectory_path}")
        
        self.f = h5py.File(trajectory_path, 'r')
        
        # Load joint data
        self.names = self.f['state/joint/joint_name'][:].tolist()[0]    
        self.angles = self.f['state/joint/joint_angle_rad'][:]
        self.cartesian = self.f['state/cartesian/cartesian_pose'][:]
        
        # Get action data if available
        self.has_action = 'action' in self.f.keys()
        if self.has_action:
            self.cartesian_action = self.f['action/cartesian_pose'][:]
        
        # Load camera data - create a new dictionary to avoid modification during iteration
        new_image_cache = {}
        for camera_name in self.f['observation'].keys():
            new_image_cache[camera_name] = self.f[f'observation/{camera_name}/image/camera_rgb'][:]
        
        # Safely update the image cache
        self.image_cache = new_image_cache
        
        # Set frame count
        self.total_frames = self.f['state/joint/joint_angle_rad'].shape[0]
        
        # Reset GUI elements if they exist
        if hasattr(self, 'slider_handle'):
            # Temporarily disable the slider update callback to prevent race conditions
            old_callbacks = []
            if hasattr(self.slider_handle, '_callbacks'):
                old_callbacks = self.slider_handle._callbacks.copy()
                self.slider_handle._callbacks = []
            
            self.slider_handle.value = 0
            self.slider_handle.max = self.total_frames - 1
            
            # Restore slider callbacks
            if old_callbacks:
                self.slider_handle._callbacks = old_callbacks
        
        # Update the camera display handles
        # self._update_camera_handles()
        
        # Resume playback if it was previously playing
        if was_playing:
            self.play = True
            if hasattr(self, 'play_button') and hasattr(self, 'pause_button'):
                self.play_button.visible = False
                self.pause_button.visible = True
    
    def _setup_viser_scene(self):
        """Setup the viser scene with frames and visualization elements."""
        # Setup base URDF visualization
        self.urdf_vis = self.yumi.urdf_vis
        
        # Add frames for visualization
        self.tf_left_frame = self.server.scene.add_frame(
            "tf_left",
            axes_length=0.5 * 0.2,
            axes_radius=0.01 * 0.2,
            origin_radius=0.1 * 0.2,
        )
        
        self.tf_right_frame = self.server.scene.add_frame(
            "tf_right",
            axes_length=0.5 * 0.2,
            axes_radius=0.01 * 0.2,
            origin_radius=0.1 * 0.2,
        )
        
        # Add action frames if action data is available
        if self.has_action:
            self.tf_left_action_frame = self.server.scene.add_frame(
                "tf_left_action",
                axes_length=0.5 * 0.2,
                axes_radius=0.01 * 0.2,
                origin_radius=0.1 * 0.2,
            )
            self.tf_right_action_frame = self.server.scene.add_frame(
                "tf_right_action",
                axes_length=0.5 * 0.2,
                axes_radius=0.01 * 0.2,
                origin_radius=0.1 * 0.2,
            )
        
        # Initialize the positions
        self.tf_left_frame.position = self.cartesian[0, 4:7]
        self.tf_left_frame.wxyz = self.cartesian[0, 0:4]
        self.tf_right_frame.position = self.cartesian[0, 11:14]
        self.tf_right_frame.wxyz = self.cartesian[0, 7:11]
        
        if self.has_action:
            self.tf_left_action_frame.position = self.cartesian_action[0, 4:7]
            self.tf_left_action_frame.wxyz = self.cartesian_action[0, 0:4]
            self.tf_right_action_frame.position = self.cartesian_action[0, 11:14]
            self.tf_right_action_frame.wxyz = self.cartesian_action[0, 7:11]
        
        # Initialize the robot configuration
        initial_config = self.names_angles_to_dict(self.names, self.angles, 0)
        self.urdf_vis.update_cfg(initial_config)
    
    def _setup_viser_gui(self):
        """Setup the GUI controls for the visualization."""
        # Trajectory selection dropdown
        with self.server.gui.add_folder("Trajectory Selection"):
            # Use full paths as values but only names for display
            trajectory_names = [str(p.name) for p in self.trajectories]
            trajectory_paths = [str(p) for p in self.trajectories]
            
            # Create a dropdown with full paths as values but only showing filenames
            self.trajectory_selector = self.server.gui.add_dropdown(
                "Select Trajectory",
                options=dict(zip(trajectory_names, trajectory_paths)),
                initial_value=str(self.current_trajectory)
            )
            
            # Add path display
            self.traj_path_display = self.server.gui.add_text(
                "Current Path", 
                f"Path: {self.current_trajectory}"
            )
            
            # Add navigation buttons for trajectories
            with self.server.gui.add_folder("Navigation"):
                self.prev_traj_button = self.server.gui.add_button(
                    label="Previous Trajectory", 
                    icon=viser.Icon.CHEVRON_LEFT
                )
                self.next_traj_button = self.server.gui.add_button(
                    label="Next Trajectory", 
                    icon=viser.Icon.CHEVRON_RIGHT
                )
                self.trajectory_info = self.server.gui.add_text(
                    "Info", 
                    f"Trajectory {self.trajectories.index(self.current_trajectory) + 1} of {len(self.trajectories)}"
                )
        
        # Playback controls
        with self.server.gui.add_folder("Controls"):
            self.play_button = self.server.gui.add_button(label="Play", icon=viser.Icon.PLAYER_PLAY_FILLED)
            self.pause_button = self.server.gui.add_button(label="Pause", icon=viser.Icon.PLAYER_PAUSE_FILLED, visible=False)
            self.next_button = self.server.gui.add_button(label="Step Forward", icon=viser.Icon.ARROW_BIG_RIGHT_FILLED)
            self.prev_button = self.server.gui.add_button(label="Step Back", icon=viser.Icon.ARROW_BIG_LEFT_FILLED)
        
        # Frame slider
        self.slider_handle = self.server.gui.add_slider(
            "Data Entry Index", min=0, max=self.total_frames-1, step=1, initial_value=0
        )
        
        # Initialize image handles collection
        self.viser_img_handles = {}
        
        # Add the camera images
        self._update_camera_handles()
        
        # Add state information
        with self.server.gui.add_folder("State"):
            # Add more state information if needed
            pass
            
        # Add action information if available
        if self.has_action:
            with self.server.gui.add_folder("Action"):
                # Add action information if needed
                pass
        
        # Add a new folder for video export functionality
        with self.server.gui.add_folder("Export Video"):
            self.resolution_scale = self.server.gui.add_dropdown(
                "Resolution Scale",
                options=("Original (1x)", "Half (1/2)", "Quarter (1/4)", "Eighth (1/8)",
                initial_value="Original (1x)"
            )
            self.fps_selector = self.server.gui.add_number("FPS", 24, min=1, max=60, step=1)
            self.save_video_button = self.server.gui.add_button(
                label="Save Cameras to MP4", 
                icon=viser.Icon.VIDEO
            )
            self.video_status = self.server.gui.add_text("Status", "Status: Ready")
        
        # Register event handlers
        @self.trajectory_selector.on_update
        def _(_) -> None:
            try:
                selected_path_str = self.trajectory_selector.value
                selected_path = Path(selected_path_str)
                
                # Only reload if it's a different trajectory
                if selected_path != self.current_trajectory:
                    self._load_trajectory(selected_path)
                    # Update the frame display but inside a try-except to catch potential errors
                    try:
                        self._update_frame(0)
                    except Exception as e:
                        print(f"Error updating initial frame: {e}")
                    
                    # Update trajectory info displays
                    self.trajectory_info.value = f"Trajectory {self.trajectories.index(self.current_trajectory) + 1} of {len(self.trajectories)}"
                    self.traj_path_display.value = f"Path: {self.current_trajectory}"
            except Exception as e:
                print(f"Error switching trajectories: {e}")
        
        @self.prev_traj_button.on_click
        def _(_) -> None:
            try:
                current_idx = self.trajectories.index(self.current_trajectory)
                # Wrap around to last trajectory if at the beginning
                new_idx = (current_idx - 1) % len(self.trajectories)
                new_trajectory = self.trajectories[new_idx]
                self.trajectory_selector.value = str(new_trajectory)
            except Exception as e:
                print(f"Error navigating to previous trajectory: {e}")
            
        @self.next_traj_button.on_click
        def _(_) -> None:
            try:
                current_idx = self.trajectories.index(self.current_trajectory)
                # Wrap around to first trajectory if at the end
                new_idx = (current_idx + 1) % len(self.trajectories)
                new_trajectory = self.trajectories[new_idx]
                self.trajectory_selector.value = str(new_trajectory)
            except Exception as e:
                print(f"Error navigating to next trajectory: {e}")
        
        @self.next_button.on_click
        def _(_) -> None:
            self.slider_handle.value = min(self.slider_handle.value + 1, self.total_frames - 1)
        
        @self.prev_button.on_click
        def _(_) -> None:
            self.slider_handle.value = max(self.slider_handle.value - 1, 0)
        
        @self.play_button.on_click
        def _(_) -> None:
            self.play = True
            self.play_button.visible = False
            self.pause_button.visible = True
        
        @self.pause_button.on_click
        def _(_) -> None:
            self.play = False
            self.play_button.visible = True
            self.pause_button.visible = False
        
        @self.slider_handle.on_update
        def _(_) -> None:
            self._update_frame(self.slider_handle.value)
        
        @self.save_video_button.on_click
        def _(_) -> None:
            if self.resolution_scale.value == "Original (1x)":
                scale = 1
            elif self.resolution_scale.value == "Half (1/2)":
                scale = 2
            elif self.resolution_scale.value == "Quarter (1/4)":
                scale = 4
            elif self.resolution_scale.value == "Eighth (1/8)":
                scale = 8
            self.save_cameras_to_mp4(scale_factor=scale, fps=self.fps_selector.value)
    
    def _update_camera_handles(self):
        """Update the camera handles based on the current image cache."""
        # Create a safe copy of the keys to avoid modification during iteration
        current_cameras = list(self.image_cache.keys())
        
        # If we don't have image handles yet, create them
        if not hasattr(self, 'viser_img_handles'):
            self.viser_img_handles = {}
        
        # First, update or create handles for current cameras
        # print(self.viser_img_handles.keys())
        for camera_name in current_cameras:
            if camera_name in self.viser_img_handles:
                # Update existing handle
                try:
                    self.viser_img_handles[camera_name].image = self.image_cache[camera_name][0]
                except Exception as e:
                    print(f"Error updating image for camera {camera_name}: {e}")
            else:
                # Create new handle if it doesn't exist
                try:
                    with self.server.gui.add_folder("Observation", visible=True):
                        self.viser_img_handles[camera_name] = self.server.gui.add_image(
                            image=self.image_cache[camera_name][0],
                            label=camera_name
                        )
                except Exception as e:
                    print(f"Error creating image handle for camera {camera_name}: {e}")
    
    def _update_frame(self, frame_idx: int):
        """Update the visualization based on the current frame index."""
        try:
            # Update joint configuration
            config = self.names_angles_to_dict(self.names, self.angles, frame_idx)
            self.urdf_vis.update_cfg(config)
            
            # Update position frames
            self.tf_left_frame.position = self.cartesian[frame_idx, 4:7]
            self.tf_left_frame.wxyz = self.cartesian[frame_idx, 0:4]
            self.tf_right_frame.position = self.cartesian[frame_idx, 11:14]
            self.tf_right_frame.wxyz = self.cartesian[frame_idx, 7:11]
            
            # Update action frames if available
            if self.has_action:
                self.tf_left_action_frame.position = self.cartesian_action[frame_idx, 4:7]
                self.tf_left_action_frame.wxyz = self.cartesian_action[frame_idx, 0:4]
                self.tf_right_action_frame.position = self.cartesian_action[frame_idx, 11:14]
                self.tf_right_action_frame.wxyz = self.cartesian_action[frame_idx, 7:11]
            
            # Update camera images - use a copy of the keys to avoid modification during iteration
            camera_names = list(self.image_cache.keys())
            for camera_name in camera_names:
                if camera_name in self.viser_img_handles:
                    self.viser_img_handles[camera_name].image = self.image_cache[camera_name][frame_idx]
        except Exception as e:
            print(f"Error updating frame {frame_idx}: {e}")
    
    def names_angles_to_dict(self, names, angles, idx=0):
        """Convert joint names and angles to a dictionary."""
        config = {}
        for i in range(len(names)):
            config[names[i].decode("utf-8")] = angles[idx, i]
        return config
    
    def save_cameras_to_mp4(self, scale_factor=1, fps=24.0):
        """Save camera frames to MP4 videos with optional resolution scaling."""
        try:
            self.video_status.value = "Status: Saving videos..."
            
            # Create a descriptive name for the video file
            trajectory_name = self.current_trajectory.stem
            
            # Make sure we have images to save
            if not hasattr(self, 'image_cache') or not self.image_cache:
                self.video_status.value = "Status: No images to save!"
                return
            
            # Get a list of camera names
            camera_names = list(self.image_cache.keys())
            if not camera_names:
                self.video_status.value = "Status: No cameras found!"
                return
            
            # Create output directory if it doesn't exist
            output_dir = self.current_trajectory.parent / "exported_videos"
            os.makedirs(output_dir, exist_ok=True)
            
            saved_files = []
            
            # Process each camera
            for camera_name in camera_names:
                # Get frames for this camera
                frames = self.image_cache[camera_name]
                if frames.shape[0] == 0:
                    continue
                
                # Get original dimensions
                height, width, channels = frames[0].shape
                
                # Calculate new dimensions based on scale factor
                new_width = width // scale_factor
                new_height = height // scale_factor
                
                # Create output filename
                output_file = f"{trajectory_name}_{camera_name}_scale{scale_factor}x.mp4"
                output_path = os.path.join(output_dir, output_file)
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
                
                # Write frames to video
                for frame in frames:
                    # Resize the frame if needed
                    if scale_factor > 1:
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # OpenCV uses BGR format, but our frames are likely RGB
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_frame)
                
                # Release video writer
                video_writer.release()
                saved_files.append(output_path)
                
                print(f"Saved video to {output_path}")
            
            if saved_files:
                self.video_status.value = f"Status: Saved {len(saved_files)} videos to {output_dir}"
            else:
                self.video_status.value = "Status: No videos were saved"
                
        except Exception as e:
            self.video_status.value = f"Status: Error - {str(e)}"
            print(f"Error saving videos: {e}")
    
    def run(self):
        """Run the main visualization loop."""
        try:
            print(f"Viewer started with {len(self.trajectories)} trajectories loaded.")
            print(f"Current trajectory: {self.current_trajectory}")
            
            while True:
                try:
                    if self.play and hasattr(self, 'slider_handle') and hasattr(self, 'total_frames'):
                        # Protect against race conditions
                        if self.total_frames > 0:
                            new_value = (self.slider_handle.value + 1) % self.total_frames
                            
                            # Only update if different (saves callbacks)
                            if new_value != self.slider_handle.value:
                                # Temporarily disable callbacks
                                old_callbacks = []
                                if hasattr(self.slider_handle, '_callbacks'):
                                    old_callbacks = self.slider_handle._callbacks.copy()
                                    self.slider_handle._callbacks = []
                                
                                # Update value
                                self.slider_handle.value = new_value
                                
                                # Manually call the update frame function
                                self._update_frame(new_value)
                                
                                # Restore callbacks
                                if old_callbacks:
                                    self.slider_handle._callbacks = old_callbacks
                            time.sleep(0.05)
                except Exception as e:
                    print(f"Error in playback loop: {e}")
                    # If we encounter an error, pause playback to prevent error spam
                    self.play = False
                    if hasattr(self, 'play_button') and hasattr(self, 'pause_button'):
                        self.play_button.visible = True
                        self.pause_button.visible = False
                

        except KeyboardInterrupt:
            print("Visualization stopped by user.")
        finally:
            if hasattr(self, 'f') and self.f:
                self.f.close()
                print("File resources cleaned up.")


def main(
    trajectory_dir: str = '/mnt/hard-drive/success/pick_up_the_tiger_042725',
):
    """Main function to launch the trajectory viewer with a directory of H5 files."""
    viewer = H5TrajectoryViewer(trajectory_dir)
    viewer.run()


if __name__ == "__main__":
    tyro.cli(main)