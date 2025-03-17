from yumi_realtime.base import YuMiBaseInterface, TransformHandle
from loguru import logger
import tyro
import jax.numpy as jnp
import numpy as onp
import time
import jaxlie
import h5py
import os

class YuMiRExtrinsics(YuMiBaseInterface):
    """YuMi class to check extrinsics calibration."""
    
    def __init__(self, *args, camera_pose_dir="camera_poses", **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_pose_dir = camera_pose_dir
        os.makedirs(camera_pose_dir, exist_ok=True)
        
        self.chessboard_frame = self.server.scene.add_frame(
            name="/chessboard",
            axes_length=0.1,
            axes_radius=0.005,
            position=onp.array([0.537, 0, 0.005]),
            wxyz=onp.array([1, 0, 0, 0]),
        )
        left_camera2cb = onp.linalg.inv(onp.load("top_camera_transform.npy"))
        right_camera2cb = onp.linalg.inv(onp.load("ego_camera_transform.npy"))

        self.left_camera_frame = self.server.scene.add_frame(
            name="/chessboard/top_camera",
            axes_length=0.1,
            axes_radius=0.005,
            position=left_camera2cb[:3, 3],
            wxyz=jaxlie.SO3.from_matrix(left_camera2cb[:3, :3]).wxyz,
        )
        
        self.right_camera_frame = self.server.scene.add_frame(
            name="/chessboard/ego_camera",
            axes_length=0.1,
            axes_radius=0.005,
            position=right_camera2cb[:3, 3],
            wxyz=jaxlie.SO3.from_matrix(right_camera2cb[:3, :3]).wxyz,
        )
        
        chessboard2world_SE3 = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(self.chessboard_frame.wxyz), self.chessboard_frame.position)
        left_camera2chessboard_SE3 = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(self.left_camera_frame.wxyz), self.left_camera_frame.position)
        right_camera2chessboard_SE3 = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(self.right_camera_frame.wxyz), self.right_camera_frame.position)

        left_camera2world_SE3 = chessboard2world_SE3 @ left_camera2chessboard_SE3
        right_camera2world_SE3 = chessboard2world_SE3 @ right_camera2chessboard_SE3
        
        self.left_camera_frame = self.server.scene.add_frame(
            name="/left_camera",
            axes_length=0.1,
            axes_radius=0.005,
            position=left_camera2world_SE3.translation(),
            wxyz=left_camera2world_SE3.rotation().wxyz,
        )
        self.right_camera_frame = self.server.scene.add_frame(
            name="/right_camera",
            axes_length=0.1,
            axes_radius=0.005,
            position=right_camera2world_SE3.translation(),
            wxyz=right_camera2world_SE3.rotation().wxyz,
        )
        
        self.save_camera_pose(self.left_camera_frame.position, 
                            self.left_camera_frame.wxyz)
        self.save_camera_pose(self.right_camera_frame.position,
                            self.right_camera_frame.wxyz)
        
        
    def save_camera_pose(self, position, wxyz):
        """Save new camera pose to file"""
        pose = onp.array([*position, *wxyz])
        camera_poses_file = os.path.join(self.camera_pose_dir, "camera_poses.h5")
        
        with h5py.File(camera_poses_file, 'a') as f:
            if 'poses' not in f:
                f.create_dataset('poses', data=pose[onp.newaxis, :],
                               maxshape=(None, 7), chunks=True)
            else:
                dset = f['poses']
                current_size = dset.shape[0]
                dset.resize(current_size + 1, axis=0)
                dset[current_size] = pose
    
    def run(self):
        while True:
            time.sleep(1)
            # self.server.step()

def main(): 
    yumi_extr = YuMiRExtrinsics().run()
    
if __name__ == "__main__":
    tyro.cli(main)