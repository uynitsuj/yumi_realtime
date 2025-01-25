import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation

class ArucoExtrinsicsCalibrator:
    def __init__(self, device_id=0, width=1920, height=1080, fps=15, 
                 aruco_dict_type=cv2.aruco.DICT_6X6_250,
                 marker_size_m=0.05, output_file="transforms.npy"):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.marker_size = marker_size_m
        self.output_file = output_file
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        
        # Statistics
        self.total_frames = 0
        self.valid_frames = 0
        
        # Store transformations
        self.transforms = []
        
    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids
        
    def process_frame(self, frame, camera_matrix, dist_coeffs):
        self.total_frames += 1
        display_frame = frame.copy()
        
        corners, ids = self.detect_markers(frame)
        valid = len(corners) > 0 if corners is not None else False
        
        if valid:
            self.valid_frames += 1
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, camera_matrix, dist_coeffs)
            
            # Store SE3 transforms
            for rvec, tvec in zip(rvecs, tvecs):
                R = Rotation.from_rotvec(rvec[0]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec[0]
                self.transforms.append(T)
            
            # Draw axes
            for i in range(len(ids)):
                cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs, 
                                rvecs[i], tvecs[i], self.marker_size/2)
            
            status_color = (0, 255, 0)
        else:
            status_color = (0, 0, 255)
            
        # Add status text
        cv2.putText(display_frame, f"Stored transforms: {len(self.transforms)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Current Frame: {'Valid' if valid else 'Invalid'}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display_frame, f"Success Rate: {self.valid_frames/self.total_frames*100:.1f}%", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return display_frame, valid
        
    def compute_mean_transform(self):
        if not self.transforms:
            return None
            
        # Convert to numpy array
        transforms = np.array(self.transforms)
        
        # Extract rotations and translations
        Rs = transforms[:, :3, :3]
        ts = transforms[:, :3, 3]
        
        # Average translations
        mean_t = np.mean(ts, axis=0)
        
        # Average rotations using quaternions
        quats = [Rotation.from_matrix(R).as_quat() for R in Rs]
        mean_quat = np.mean(quats, axis=0)
        mean_quat /= np.linalg.norm(mean_quat)  # Normalize
        mean_R = Rotation.from_quat(mean_quat).as_matrix()
        
        # Construct mean transform
        mean_T = np.eye(4)
        mean_T[:3, :3] = mean_R
        mean_T[:3, 3] = mean_t
        
        return mean_T
        
    def run(self, camera_matrix, dist_coeffs):
        print("Press 's' to save mean transform")
        print("Press 'r' to reset collected transforms")
        print("Press 'q' to quit")
        
        while True:
            current_time = time.time()
            if (current_time - self.last_frame_time) < self.frame_interval:
                continue
                
            self.last_frame_time = current_time
            
            ret, frame = self.cap.read()
            if not ret:
                break
                
            display_frame, is_valid = self.process_frame(frame, camera_matrix, dist_coeffs)
            cv2.imshow('ArUco Extrinsics Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                mean_transform = self.compute_mean_transform()
                if mean_transform is not None:
                    np.save(self.output_file, mean_transform)
                    print(f"Saved mean transform to {self.output_file}")
                    print("Mean transform:")
                    print(mean_transform)
                else:
                    print("No transforms collected yet")
            elif key == ord('r'):
                self.transforms = []
                self.total_frames = 0
                self.valid_frames = 0
                print("Reset transforms")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage with dummy camera matrix and distortion coefficients
    # camera_matrix = np.array([[686.53174572,   0.,         966.99788553],
    # [  0.,         636.40382904, 512.00345383],
    # [  0.,           0.,           1.        ]])
    # dist_coeffs = np.array([[ 6.49342319e-02, -4.03104967e-01, -1.89174280e-02,  2.86095171e-04,   1.29272219e+00]])
    
    camera_matrix = np.array([[1.11051899e+03, 0.00000000e+00, 9.76018042e+02],
    [0.00000000e+00, 1.11032898e+03, 5.33269878e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist_coeffs = np.array([[ 0.16449587, -0.36449848, -0.00538376, 0.00240608, -0.09279666]])

    try:
        calibrator = ArucoExtrinsicsCalibrator(
            device_id=0,
            marker_size_m=0.1552,  # 15.52cm marker
            output_file="left_camera_transform.npy"
        )
        calibrator.run(camera_matrix, dist_coeffs)
    except Exception as e:
        print(f"Error: {str(e)}")