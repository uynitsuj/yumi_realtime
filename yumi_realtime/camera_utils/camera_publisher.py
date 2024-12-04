import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

class CameraPublisher:
    def __init__(
        self, 
        device_id: int=0,
        name: str='camera_0',
        image_height: int=480,
        image_width: int=848,
        fps: int=30,
        init_node: bool=False
        ):
        
        if init_node:
            rospy.init_node('camera_publisher', anonymous=True)
        
        self.name = name
        
        # Initialize the CvBridge class
        self.bridge = CvBridge()
        
        # Initialize the camera
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera!")
            return
        
        self.height = image_height 
        self.width = image_width 
        self.fps = fps

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Create publishers
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        
        # Set up the publishing rate
        self.rate = rospy.Rate(self.fps)
        
        rospy.loginfo(f"Started camera publisher - Resolution: {self.width}x{self.height}, FPS: {self.fps}")

    def run(self):
        while not rospy.is_shutdown():
            # Read frame from the camera
            ret, frame = self.cap.read()
            
            if ret:
                try:
                    # Convert the OpenCV image to a ROS image message
                    ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    
                    # Add header timestamp
                    ros_image.header.stamp = rospy.Time.now()
                    ros_image.header.frame_id = self.name
                    
                    # Publish the image
                    self.image_pub.publish(ros_image)
                    
                except Exception as e:
                    rospy.logerr(f"Failed to publish image: {str(e)}")
            
            self.rate.sleep()

    def __del__(self):
        """Cleanup camera resources"""
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == '__main__':
    try:
        camera_pub = CameraPublisher(init_node=True)
        camera_pub.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'camera_pub' in locals():
            del camera_pub