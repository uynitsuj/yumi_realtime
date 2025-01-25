import cv2
import numpy as np
import time

# Checkerboard parameters
CHECKERBOARD = (6, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class CameraCalibrator:
    def __init__(self, device_id=0, width=1920, height=1080, fps=2, square_size_m=.233/9):
        # Initialize camera
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Calibration data storage
        self.objpoints = []
        self.imgpoints = []
        
        # Create 3D points
        self.objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= square_size_m
        
        # Frame rate control
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        
        # Statistics
        self.total_frames = 0
        self.valid_frames = 0
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

    def is_valid_pattern(self, frame):
        """Check if frame contains a valid checkerboard pattern"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            return True, gray, corners2
        return False, None, None

    def process_frame(self, frame):
        """Process a frame and return display frame with annotations"""
        self.total_frames += 1
        display_frame = frame.copy()
        
        valid, gray, corners = self.is_valid_pattern(frame)
        
        if valid:
            self.valid_frames += 1
            
            # Store calibration points
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            
            # Draw corners on display frame
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, True)
            status_color = (0, 255, 0)  # Green for valid pattern
        else:
            status_color = (0, 0, 255)  # Red for invalid pattern
        
        # Add status text
        cv2.putText(
            display_frame,
            f"Valid Patterns: {len(self.imgpoints)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            display_frame,
            f"Current Frame: {'Valid' if valid else 'Invalid'}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,
            status_color,
            2
        )
        
        cv2.putText(
            display_frame,
            f"Success Rate: {self.valid_frames/self.total_frames*100:.1f}%", 
            (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,
            (255, 255, 0),
            2
        )
        
        return display_frame, valid

    def calibrate(self):
        if len(self.objpoints) > 5:  # Need at least 5 patterns for good calibration
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints,
                self.imgpoints,
                self.current_resolution[::-1],  # Swap width/height
                None,
                None
            )
            
            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                self.rvecs = rvecs
                self.tvecs = tvecs
                
                # Calculate reprojection error
                mean_error = 0
                for i in range(len(self.objpoints)):
                    imgpoints2, _ = cv2.projectPoints(
                        self.objpoints[i],
                        rvecs[i],
                        tvecs[i],
                        mtx,
                        dist
                    )
                    error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                    mean_error += error
                
                print("\nCalibration Results:")
                print("Camera matrix:")
                print(self.camera_matrix)
                print("\nDistortion coefficients:")
                print(self.dist_coeffs)
                print("\nTranslation Vectors:")
                print(self.tvecs)
                print("\nRotation Vectors:")
                print(self.rvecs)
                print(f"\nTotal reprojection error: {mean_error/len(self.objpoints)}")
                print(f"\nUsed {len(self.objpoints)} valid frames out of {self.total_frames} total frames")
                print(f"Frame validation rate: {self.valid_frames/self.total_frames*100:.1f}%")
                
                return True
        return False

    def run(self):
        print("Press 'c' to perform calibration")
        print("Press 'r' to reset collected patterns")
        print("Press 'q' to quit")
        
        while True:
            current_time = time.time()
            
            # Control frame rate
            if (current_time - self.last_frame_time) < self.frame_interval:
                continue
                
            self.last_frame_time = current_time
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            self.current_resolution = frame.shape[:2]  # Store for calibration
            
            # Process frame
            display_frame, is_valid = self.process_frame(frame)
            
            # Show frame
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if self.calibrate():
                    print("Calibration successful!")
                else:
                    print("Need at least 5 different patterns for calibration")
            elif key == ord('r'):
                self.objpoints = []
                self.imgpoints = []
                self.total_frames = 0
                self.valid_frames = 0
                print("Reset calibration points")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        calibrator = CameraCalibrator(device_id=0, fps=15)  # 2 FPS for stable processing
        calibrator.run()
    except Exception as e:
        print(f"Error: {str(e)}")