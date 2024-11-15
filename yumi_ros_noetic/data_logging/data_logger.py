#!/usr/bin/env python

import rospy
import h5py
import numpy as np
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import message_filters
from datetime import datetime
import os

class DataLogger:
    def __init__(self):
        rospy.init_node('data_logger', anonymous=True)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.expanduser('~/ros_logs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create HDF5 file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = os.path.join(self.output_dir, f'robot_data_{timestamp}.h5')
        self.file = h5py.File(self.filename, 'w')
        
        # Create datasets in the HDF5 file
        self.image_dataset = self.file.create_group('images')
        self.joint_dataset = self.file.create_group('joint_states')
        
        # Initialize message filters for synchronization
        self.image_sub = message_filters.Subscriber('/camera/image_raw', Image)
        self.joint_sub = message_filters.Subscriber('/joint_states', JointState)
        
        # Time synchronizer
        # Adjust queue_size and slop as needed for your setup
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.joint_sub],
            queue_size=10,
            slop=0.1  # 100ms time difference tolerance
        )
        self.ts.registerCallback(self.callback)
        
        self.count = 0
        rospy.loginfo("Data logger initialized. Saving to: " + self.filename)

    def callback(self, image_msg, joint_msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # Create new datasets for this timestamp
            timestamp = str(image_msg.header.stamp.to_sec())
            
            # Save image
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
                'effort': joint_msg.effort,
                'name': [n.encode('utf-8') for n in joint_msg.name]  # Convert strings to bytes for HDF5
            }
            
            joint_group = self.joint_dataset.create_group(timestamp)
            for key, value in joint_data.items():
                joint_group.create_dataset(key, data=value)
            
            self.count += 1
            if self.count % 100 == 0:
                rospy.loginfo(f"Logged {self.count} synchronized observations")
                
        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

    def __del__(self):
        """Cleanup when the node is shut down"""
        if hasattr(self, 'file'):
            self.file.close()
            rospy.loginfo("HDF5 file closed")

if __name__ == '__main__':
    try:
        logger = DataLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'logger' in locals():
            del logger  # Ensure proper cleanup