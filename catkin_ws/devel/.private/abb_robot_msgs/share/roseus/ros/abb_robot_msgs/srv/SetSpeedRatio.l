;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::SetSpeedRatio)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'SetSpeedRatio (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SETSPEEDRATIO")
  (make-package "ABB_ROBOT_MSGS::SETSPEEDRATIO"))
(unless (find-package "ABB_ROBOT_MSGS::SETSPEEDRATIOREQUEST")
  (make-package "ABB_ROBOT_MSGS::SETSPEEDRATIOREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::SETSPEEDRATIORESPONSE")
  (make-package "ABB_ROBOT_MSGS::SETSPEEDRATIORESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::SetSpeedRatioRequest
  :super ros::object
  :slots (_speed_ratio ))

(defmethod abb_robot_msgs::SetSpeedRatioRequest
  (:init
   (&key
    ((:speed_ratio __speed_ratio) 0)
    )
   (send-super :init)
   (setq _speed_ratio (round __speed_ratio))
   self)
  (:speed_ratio
   (&optional __speed_ratio)
   (if __speed_ratio (setq _speed_ratio __speed_ratio)) _speed_ratio)
  (:serialization-length
   ()
   (+
    ;; uint8 _speed_ratio
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _speed_ratio
       (write-byte _speed_ratio s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _speed_ratio
     (setq _speed_ratio (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(defclass abb_robot_msgs::SetSpeedRatioResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_robot_msgs::SetSpeedRatioResponse
  (:init
   (&key
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::SetSpeedRatio
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::SetSpeedRatio :md5sum-) "a37a98881c87fe9569621a0efb03c471")
(setf (get abb_robot_msgs::SetSpeedRatio :datatype-) "abb_robot_msgs/SetSpeedRatio")
(setf (get abb_robot_msgs::SetSpeedRatio :request) abb_robot_msgs::SetSpeedRatioRequest)
(setf (get abb_robot_msgs::SetSpeedRatio :response) abb_robot_msgs::SetSpeedRatioResponse)

(defmethod abb_robot_msgs::SetSpeedRatioRequest
  (:response () (instance abb_robot_msgs::SetSpeedRatioResponse :init)))

(setf (get abb_robot_msgs::SetSpeedRatioRequest :md5sum-) "a37a98881c87fe9569621a0efb03c471")
(setf (get abb_robot_msgs::SetSpeedRatioRequest :datatype-) "abb_robot_msgs/SetSpeedRatioRequest")
(setf (get abb_robot_msgs::SetSpeedRatioRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the
#   speed ratio for an ABB robot controller system's RAPID motions
#   (e.g. MoveAbsJ and MoveL).
#
# Note:
#   The speed ratio must be an integer in the range [0, 100] (i.e. inclusive).
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# New speed ratio.
uint8 speed_ratio

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_robot_msgs::SetSpeedRatioResponse :md5sum-) "a37a98881c87fe9569621a0efb03c471")
(setf (get abb_robot_msgs::SetSpeedRatioResponse :datatype-) "abb_robot_msgs/SetSpeedRatioResponse")
(setf (get abb_robot_msgs::SetSpeedRatioResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the
#   speed ratio for an ABB robot controller system's RAPID motions
#   (e.g. MoveAbsJ and MoveL).
#
# Note:
#   The speed ratio must be an integer in the range [0, 100] (i.e. inclusive).
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# New speed ratio.
uint8 speed_ratio

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_robot_msgs/SetSpeedRatio "a37a98881c87fe9569621a0efb03c471")


