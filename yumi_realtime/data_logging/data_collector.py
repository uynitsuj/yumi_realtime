#!/usr/bin/env python

import rospy
import h5py
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Empty, EmptyResponse
from cv_bridge import CvBridge
from datetime import datetime
import os
from threading import Lock
from jaxmp import JaxKinTree
from jaxmp.extras.urdf_loader import load_urdf
import jax.numpy as jnp
import numpy as onp
import jaxlie
import shutil
from vr_policy.msg import VRPolicyAction, OculusData

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../trajectories/data")

class DataCollector:
    def __init__(
        self,
        init_node: bool = False,
        save_data: bool = True,
        save_traj_dir: str = None,
        image_height: int = 480, 
        image_width: int = 848,
        task_name: str = 'example_task1'
        ):
        
        if init_node:
            rospy.init_node('data_logger')
            
        self.image_resolution = (image_height, image_width)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Create log directories if they don't exist
        if save_traj_dir is None:
            save_traj_dir = data_dir
        self.success_logdir = os.path.join(save_traj_dir, "success", task_name)
        self.failure_logdir = os.path.join(save_traj_dir, "failure", task_name)
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
        self.save_data = save_data
        
        # Initialize file handles as None
        self.file = None
        self.image_dataset = None
        self.joint_angle_dataset = None
        self.joint_velocity_dataset = None
        
        # Threading lock for file operations
        self.file_lock = Lock()
        
        # Forward kinematics
        self.urdf = load_urdf("yumi", None)
        self.kin = JaxKinTree.from_urdf(self.urdf)
        self.base_pose = jaxlie.SE3.identity()
        
        # Streaming control
        self.is_recording = False
        
        self._first_line = True
        
        self.current_image = None
        
        self.max_buffer_length = 80
        self.current_joint_buffer = []
        self.current_action_L_buffer = []
        self.current_action_R_buffer = []

        self.max_buffer_length_egm = 500
        self.current_joint_buffer_egm = []

        # Initialize message filters for synchronization
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.joint_sub = rospy.Subscriber('/yumi/combined/joint_states', JointState, self.joint_callback)
        self.vr_policy_L_sub = rospy.Subscriber('/vr_policy/control_l', VRPolicyAction, self.vr_policy_L_callback)
        self.vr_policy_R_sub = rospy.Subscriber('/vr_policy/control_r', VRPolicyAction, self.vr_policy_R_callback)

        self.enable_left = False 
        self.enable_right = False

        self.joint_sub_egm = rospy.Subscriber('/yumi/egm/joint_states', JointState, self.joint_callback_egm)
        
        # Set up ROS services for control
        rospy.Service('~start_recording', Empty, self.start_recording)
        rospy.Service('~save_failure', Empty, self.save_failure)
        rospy.Service('~save_success', Empty, self.save_success)
        
        self.count = 0
        rospy.loginfo("Data logger initialized and ready to record")

    def image_callback(self, image_msg):
        if not self.is_recording:
            return
        with self.file_lock:
            if self.file is None:
                return
            active = self.enable_left or self.enable_right
            if not active:
                return 
            timestamp = image_msg.header.stamp
            target_time = timestamp.to_nsec()
            closest_joint_state = min( #retrieve from current_joint_buffer the joint state closest in time to the image timestamp
                                        self.current_joint_buffer,
                                        key=lambda joint_state: abs(joint_state.header.stamp.to_nsec()-target_time)
                                    )
            
            closest_joint_state_egm = min( #retrieve from current_joint_buffer_egm the joint state closest in time to the image timestamp
                                        self.current_joint_buffer_egm,
                                        key=lambda joint_state: abs(joint_state.header.stamp.to_nsec()-target_time)
                                    )
            
            closest_action_L = min( #retrieve from current_action_L_buffer the joint state closest in time to the image timestamp
                                        self.current_action_L_buffer,
                                        key=lambda action: abs(action.header.stamp.to_nsec()-target_time)
                                    )
            
            closest_action_R = min( #retrieve from current_action_L_buffer the joint state closest in time to the image timestamp
                                        self.current_action_R_buffer,
                                        key=lambda action: abs(action.header.stamp.to_nsec()-target_time)
                                    )

            
            rgb_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            joint_positions = list(closest_joint_state.position)

            if abs(closest_joint_state_egm.header.stamp.to_nsec() - target_time) < abs(closest_joint_state.header.stamp.to_nsec() - target_time):
                ja_dict = map_egm_joints(closest_joint_state_egm)
                joint_positions[0:14] = list(ja_dict.values())

            joint_velocities = closest_joint_state.velocity
            
            if len(joint_velocities) == 0:
                joint_velocities = [0.0]*14 # MAY NEED TO HANDLE DIFFERENTLY -- NO VEL MIGHT NOT MEAN ZERO VEL
            
            if self._first_line:
                self.joint_order_names.resize(1, axis=0)
                self.joint_order_names[0] = closest_joint_state.name
                self._first_line = False
            
            # Observation
            self.image_dataset.resize(self.count + 1, axis=0)
            self.image_dataset[self.count] = rgb_image
            
            self.image_ts_dataset.resize(self.count + 1, axis=0)
            self.image_ts_dataset[self.count] = target_time
            
            # Joint state
            self.joint_angle_dataset.resize(self.count + 1, axis=0)
            self.joint_angle_dataset[self.count] = joint_positions
            
            self.joint_velocity_dataset.resize(self.count + 1, axis=0)
            self.joint_velocity_dataset[self.count] = joint_velocities
            
            self.joint_ts_dataset.resize(self.count + 1, axis=0)
            timestamp = closest_joint_state.header.stamp
            self.joint_ts_dataset[self.count] = timestamp.to_nsec()
            
            # Cartesian state
            joints_array = jnp.array(joint_positions, dtype=jnp.float32)
            fk_frames = self.kin.forward_kinematics(joints_array)
            
            cartesian_pose = onp.zeros(14)
            for side, joint_name in [('left', 'yumi_joint_6_l'), ('right', 'yumi_joint_6_r')]:
                joint_idx = self.kin.joint_names.index(joint_name)
                T_target_world = self.base_pose @ jaxlie.SE3(fk_frames[joint_idx])
                
                if side == 'left':
                    cartesian_pose[:7] = T_target_world.wxyz_xyz
                else:
                    cartesian_pose[7:] = T_target_world.wxyz_xyz
                    
            assert not all(cartesian_pose == 0)
            
            self.cartesian_pose_dataset.resize(self.count + 1, axis=0)
            self.cartesian_pose_dataset[self.count] = cartesian_pose
            
            # Action
            cartesian_pose_action = onp.array(
                [
                    closest_action_L.transform.rotation.w,
                    closest_action_L.transform.rotation.x,
                    closest_action_L.transform.rotation.y,
                    closest_action_L.transform.rotation.z,
                    closest_action_L.transform.translation.x,
                    closest_action_L.transform.translation.y,
                    closest_action_L.transform.translation.z,
                    
                    closest_action_R.transform.rotation.w,
                    closest_action_R.transform.rotation.x,
                    closest_action_R.transform.rotation.y,
                    closest_action_R.transform.rotation.z,
                    closest_action_R.transform.translation.x,
                    closest_action_R.transform.translation.y,
                    closest_action_R.transform.translation.z,
                ]
            )
                      

            self.action_dataset.resize(self.count + 1, axis=0)
            self.action_dataset[self.count] = cartesian_pose_action

            # Flush data to disk periodically
            self.count += 1
            if self.count % 100 == 0:
                self.file.flush()
                rospy.loginfo(f"Logged {self.count} synchronized observations")
    
    def joint_callback(self, joint_msg):
        self.current_joint_buffer.append(joint_msg)
        if(len(self.current_joint_buffer)>self.max_buffer_length):
            self.current_joint_buffer.pop(0)

    def joint_callback_egm(self, joint_msg):
        self.current_joint_buffer_egm.append(joint_msg)
        if(len(self.current_joint_buffer_egm)>self.max_buffer_length_egm):
            self.current_joint_buffer_egm.pop(0)


    def vr_policy_L_callback(self, msg):
        self.enable_left = msg.enable
        self.current_action_L_buffer.append(msg.target_cartesian_pos)
        if(len(self.current_action_L_buffer)>self.max_buffer_length):
            self.current_action_L_buffer.pop(0)

    def vr_policy_R_callback(self, msg):
        self.enable_right = msg.enable
        self.current_action_R_buffer.append(msg.target_cartesian_pos)
        if(len(self.current_action_R_buffer)>self.max_buffer_length):
            self.current_action_R_buffer.pop(0)
    
    def create_new_file(self):
        """Creates a new HDF5 file with SWMR mode enabled"""
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        filename = os.path.join(self.failure_logdir, f'robot_trajectory_{timestamp}.h5')

        self.filename_suffix = f'robot_trajectory_{timestamp}.h5'
        
        # Create file with SWMR mode enabled
        self.filepath = filename
        self.file = h5py.File(filename, 'w', libver='latest')
        self.file.swmr_mode = True
        
        # Create extensible datasets with unlimited length        
        self.image_group = self.file.create_group('observation/camera/image')
        
        # Create extensible datasets with unlimited length for images
        self.image_dataset = self.image_group.create_dataset(
            'camera_rgb',
            shape=(0, *self.image_resolution, 3),
            maxshape=(None, *self.image_resolution, 3),
            chunks=(1, *self.image_resolution, 3),
            dtype='uint8',
            compression='gzip',
            compression_opts=2
        )
        
        self.image_ts_dataset = self.image_group.create_dataset(
            'timestamp_ns',
            shape=(0, 1),
            maxshape=(None, 1),
            chunks=(1, 1),
            dtype='int',
        )
        
        self.joint_group = self.file.create_group('state/joint')
        
        self.cartesian_group = self.file.create_group('state/cartesian')

        self.action_group = self.file.create_group('action')
        
        self.action_dataset = self.action_group.create_dataset(
            'cartesian_pose',
            shape=(0, 14), # [LEFT ARM] w, x, y, z, -- x, y, z + [RIGHT ARM] w, x, y, z -- x, y, z
            maxshape=(None, 14),
            chunks=(1, 14),
            dtype='float64',
        )
        
        self.cartesian_pose_dataset = self.cartesian_group.create_dataset(
            'cartesian_pose',
            shape=(0, 14), # [LEFT ARM] w, x, y, z, -- x, y, z + [RIGHT ARM] w, x, y, z -- x, y, z
            maxshape=(None, 14),
            chunks=(1, 14),
            dtype='float64',
        )
        
        self.joint_ts_dataset = self.joint_group.create_dataset(
            'timestamp_ns',
            shape=(0, 1),
            maxshape=(None, 1),
            chunks=(1, 1),
            dtype='int',
        )
        
        self.joint_order_names = self.joint_group.create_dataset(
            'joint_name',
            shape=(1, 16),     
            maxshape=(None, 16),
            chunks=(1, 16),
            dtype=h5py.string_dtype(),
        )
        self.joint_angle_dataset = self.joint_group.create_dataset(
            'joint_angle_rad',
            shape=(0, 16),     
            maxshape=(None, 16),
            chunks=(1, 16),
            dtype='float64',
        )
        
        self.joint_velocity_dataset = self.joint_group.create_dataset(
            'joint_vel_rad',
            shape=(0, 14),     
            maxshape=(None, 14),
            chunks=(1, 14),
            dtype='float64',
        )
        
        # Create metadata group
        metadata = self.file.create_group('metadata')
        metadata.create_dataset('start_time', data=timestamp)
        metadata.create_dataset('robot_model', data='yumi_irb14000')
        
        rospy.loginfo(f"Created new recording file: {filename}")
        return filename

    def start_recording(self, req):
        """ROS service handler to start recording"""
        with self.file_lock:
            if not self.is_recording:
                if self.file is not None:
                    self.file.close()
                filename = self.create_new_file()
                self.is_recording = True
                self.count = 0
                rospy.loginfo(f"Started recording to: {filename}")
        return EmptyResponse()

    def save_failure(self, req):
        """ROS service handler to stop recording"""
        with self.file_lock:
            if self.is_recording:
                if self.file is not None:
                    # Add end time metadata
                    end_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.file['metadata'].create_dataset('end_time', data=end_time.encode('utf-8'))
                    self.file['metadata'].create_dataset('total_frames', data=self.count)
                    
                    self.file.close()
                    self.file = None
                    self.image_dataset = None
                    self.joint_dataset = None
                    self._first_line = True
                
                self.is_recording = False
                rospy.loginfo("FAILED TRAJECTORY saved to {self.success_logdir}")
        return EmptyResponse()
    
    def save_success(self, req):
        """ROS service handler to stop recording"""
        with self.file_lock:
            if self.is_recording:
                if self.file is not None:
                    # Add end time metadata
                    end_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.file['metadata'].create_dataset('end_time', data=end_time.encode('utf-8'))
                    self.file['metadata'].create_dataset('total_frames', data=self.count)
                    
                    self.file.close()
                    self.file = None
                    self.image_dataset = None
                    self.joint_dataset = None
                    self._first_line = True
                
                self.is_recording = False
                os.path.join(self.success_logdir, self.filename_suffix)
                rospy.loginfo("SUCCESSFUL TRAJECTORY saved to {self.success_logdir}")
        shutil.move(self.filepath, self.success_logdir)
        
        return EmptyResponse()

    def __del__(self):
        """Cleanup when the node is shut down"""
        self.stop_recording(Empty())

if __name__ == '__main__':
    try:
        logger = DataCollector(init_node=True)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'logger' in locals():
            del logger
            
def map_egm_joints(data: JointState):
        """Remap EGM joint state order."""
        joints_real = {
                    "yumi_joint_1_r": data.position[7],
                    "yumi_joint_2_r": data.position[8],
                    "yumi_joint_7_r": data.position[9],
                    "yumi_joint_3_r": data.position[10],
                    "yumi_joint_4_r": data.position[11],
                    "yumi_joint_5_r": data.position[12],
                    "yumi_joint_6_r": data.position[13],
                    "yumi_joint_1_l": data.position[0],
                    "yumi_joint_2_l": data.position[1],
                    "yumi_joint_7_l": data.position[2],
                    "yumi_joint_3_l": data.position[3],
                    "yumi_joint_4_l": data.position[4],
                    "yumi_joint_5_l": data.position[5],
                    "yumi_joint_6_l": data.position[6],
                }
        
        return joints_real