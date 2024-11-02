;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::GetSpeedRatio)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'GetSpeedRatio (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::GETSPEEDRATIO")
  (make-package "ABB_ROBOT_MSGS::GETSPEEDRATIO"))
(unless (find-package "ABB_ROBOT_MSGS::GETSPEEDRATIOREQUEST")
  (make-package "ABB_ROBOT_MSGS::GETSPEEDRATIOREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::GETSPEEDRATIORESPONSE")
  (make-package "ABB_ROBOT_MSGS::GETSPEEDRATIORESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::GetSpeedRatioRequest
  :super ros::object
  :slots ())

(defmethod abb_robot_msgs::GetSpeedRatioRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetSpeedRatioResponse
  :super ros::object
  :slots (_speed_ratio _result_code _message ))

(defmethod abb_robot_msgs::GetSpeedRatioResponse
  (:init
   (&key
    ((:speed_ratio __speed_ratio) 0)
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _speed_ratio (round __speed_ratio))
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:speed_ratio
   (&optional __speed_ratio)
   (if __speed_ratio (setq _speed_ratio __speed_ratio)) _speed_ratio)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; uint8 _speed_ratio
    1
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _speed_ratio
       (write-byte _speed_ratio s)
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _speed_ratio
     (setq _speed_ratio (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetSpeedRatio
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::GetSpeedRatio :md5sum-) "f0753046e29c5c4aa8b7b9ff8a761daf")
(setf (get abb_robot_msgs::GetSpeedRatio :datatype-) "abb_robot_msgs/GetSpeedRatio")
(setf (get abb_robot_msgs::GetSpeedRatio :request) abb_robot_msgs::GetSpeedRatioRequest)
(setf (get abb_robot_msgs::GetSpeedRatio :response) abb_robot_msgs::GetSpeedRatioResponse)

(defmethod abb_robot_msgs::GetSpeedRatioRequest
  (:response () (instance abb_robot_msgs::GetSpeedRatioResponse :init)))

(setf (get abb_robot_msgs::GetSpeedRatioRequest :md5sum-) "f0753046e29c5c4aa8b7b9ff8a761daf")
(setf (get abb_robot_msgs::GetSpeedRatioRequest :datatype-) "abb_robot_msgs/GetSpeedRatioRequest")
(setf (get abb_robot_msgs::GetSpeedRatioRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the
#   speed ratio for an ABB robot controller system's RAPID motions
#   (e.g. MoveAbsJ and MoveL).
#
# Note:
#   The speed ratio will be an integer in the range [0, 100] (i.e. inclusive).
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Speed ratio.
uint8 speed_ratio

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_robot_msgs::GetSpeedRatioResponse :md5sum-) "f0753046e29c5c4aa8b7b9ff8a761daf")
(setf (get abb_robot_msgs::GetSpeedRatioResponse :datatype-) "abb_robot_msgs/GetSpeedRatioResponse")
(setf (get abb_robot_msgs::GetSpeedRatioResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get the
#   speed ratio for an ABB robot controller system's RAPID motions
#   (e.g. MoveAbsJ and MoveL).
#
# Note:
#   The speed ratio will be an integer in the range [0, 100] (i.e. inclusive).
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Speed ratio.
uint8 speed_ratio

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_robot_msgs/GetSpeedRatio "f0753046e29c5c4aa8b7b9ff8a761daf")


