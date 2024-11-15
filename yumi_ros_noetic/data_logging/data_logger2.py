#!/usr/bin/env python

import rospy
import h5py
import numpy as np
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Empty, EmptyResponse
from cv_bridge import CvBridge
import message_filters
from datetime import datetime
import os
from threading import Lock

class DataLogger:
    def __init__(self):
        rospy.init_node('data_logger', anonymous=True)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.expanduser('~/ros_logs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize file handles as None
        self.file = None
        self.image_dataset = None
        self.joint_dataset = None
        
        # Threading lock for file operations
        self.file_lock = Lock()
        
        # Streaming control
        self.is_recording = False
        
        # Initialize message filters for synchronization
        self.image_sub = message_filters.Subscriber('/camera/image_raw', Image)
        self.joint_sub = message_filters.Subscriber('/joint_states', JointState)
        
        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.joint_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)
        
        # Set up ROS services for control
        rospy.Service('~start_recording', Empty, self.start_recording)
        rospy.Service('~stop_recording', Empty, self.stop_recording)
        
        self.count = 0
        rospy.loginfo("Data logger initialized and ready to record")

    def create_new_file(self):
        """Creates a new HDF5 file with SWMR mode enabled"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'robot_data_{timestamp}.h5')
        
        # Create file with SWMR mode enabled
        self.file = h5py.File(filename, 'w', libver='latest')
        self.file.swmr_mode = True
        
        # Create extensible datasets with unlimited length
        self.image_dataset = self.file.create_group('images')
        self.joint_dataset = self.file.create_group('joint_states')
        
        # Create metadata group
        metadata = self.file.create_group('metadata')
        metadata.create_dataset('start_time', data=timestamp.encode('utf-8'))
        
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

    def stop_recording(self, req):
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
                
                self.is_recording = False
                rospy.loginfo("Stopped recording")
        return EmptyResponse()

    def callback(self, image_msg, joint_msg):
        """Callback for synchronized messages"""
        if not self.is_recording:
            return

        with self.file_lock:
            try:
                if self.file is None:
                    return

                # Convert ROS image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                
                # Create new datasets for this timestamp
                timestamp = str(image_msg.header.stamp.to_sec())
                
                # Save image with compression
                self.image_dataset.create_dataset(
                    timestamp,
                    data=cv_image,
                    compression="gzip",
                    compression_opts=9
                )
                
                # Save joint states
                joint_data = {
                    'position': joint_msg.position,
                    'velocity': joint_msg.velocity,
                    # 'effort': joint_msg.effort,
                    'name': [n.encode('utf-8') for n in joint_msg.name]
                }
                
                joint_group = self.joint_dataset.create_group(timestamp)
                for key, value in joint_data.items():
                    joint_group.create_dataset(key, data=value)
                
                # Flush data to disk periodically
                self.count += 1
                if self.count % 100 == 0:
                    self.file.flush()
                    rospy.loginfo(f"Logged {self.count} synchronized observations")
                    
            except Exception as e:
                rospy.logerr(f"Error in callback: {str(e)}")

    def __del__(self):
        """Cleanup when the node is shut down"""
        self.stop_recording(Empty())

if __name__ == '__main__':
    try:
        logger = DataLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'logger' in locals():
            del logger