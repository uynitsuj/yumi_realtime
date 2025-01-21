import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
import threading

class CameraPublisher:
    def __init__(
        self, 
        device_id: int=0,
        name: str='camera_0',
        image_height: int=480,
        image_width: int=848,
        fps: int=10,
        init_node: bool=False
        ):
        
        if init_node:
            rospy.init_node('camera_publisher', anonymous=True)
        
        self.name = name
        self.running = False
        self.thread = None
        
        self.bridge = CvBridge()
        
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera!")
            return
        
        self.height = image_height 
        self.width = image_width 
        self.fps = fps

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Create publishers
        self.image_pub = rospy.Publisher(f'/camera/{self.name}/image_raw', Image, queue_size=10)
        
        self.rate = rospy.Rate(self.fps)
        
        rospy.loginfo(f"Started camera publisher {self.name} - Resolution: {self.width}x{self.height}, FPS: {self.fps}")

    def _run(self):
        """Internal run method that runs in the thread"""
        while not rospy.is_shutdown() and self.running:
            ret, frame = self.cap.read()
            
            if ret:
                try:
                    ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    ros_image.header.stamp = rospy.Time.now()
                    ros_image.header.frame_id = self.name
                    self.image_pub.publish(ros_image)
                except Exception as e:
                    rospy.logerr(f"Failed to publish image from {self.name}: {str(e)}")
            
            self.rate.sleep()

    def start(self):
        """Start the camera publisher in a new thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.start()
            rospy.loginfo(f"Camera {self.name} thread started")

    def stop(self):
        """Stop the camera publisher thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
            rospy.loginfo(f"Camera {self.name} thread stopped")

    def __del__(self):
        """Cleanup camera resources"""
        self.stop()
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == '__main__':
    try:
        rospy.init_node('multi_camera_publisher')
        
        # Creates two camera publishers
        camera0 = CameraPublisher(device_id=0, name='camera_0')
        camera1 = CameraPublisher(device_id=4, name='camera_1')
        
        camera0.start()
        camera1.start()
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'camera0' in locals():
            camera0.stop()
        if 'camera1' in locals():
            camera1.stop()