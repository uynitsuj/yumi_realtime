;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::GetRobotControllerDescription)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'GetRobotControllerDescription (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::GETROBOTCONTROLLERDESCRIPTION")
  (make-package "ABB_ROBOT_MSGS::GETROBOTCONTROLLERDESCRIPTION"))
(unless (find-package "ABB_ROBOT_MSGS::GETROBOTCONTROLLERDESCRIPTIONREQUEST")
  (make-package "ABB_ROBOT_MSGS::GETROBOTCONTROLLERDESCRIPTIONREQUEST"))
(unless (find-package "ABB_ROBOT_MSGS::GETROBOTCONTROLLERDESCRIPTIONRESPONSE")
  (make-package "ABB_ROBOT_MSGS::GETROBOTCONTROLLERDESCRIPTIONRESPONSE"))

(in-package "ROS")





(defclass abb_robot_msgs::GetRobotControllerDescriptionRequest
  :super ros::object
  :slots ())

(defmethod abb_robot_msgs::GetRobotControllerDescriptionRequest
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

(defclass abb_robot_msgs::GetRobotControllerDescriptionResponse
  :super ros::object
  :slots (_description _result_code _message ))

(defmethod abb_robot_msgs::GetRobotControllerDescriptionResponse
  (:init
   (&key
    ((:description __description) "")
    ((:result_code __result_code) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _description (string __description))
   (setq _result_code (round __result_code))
   (setq _message (string __message))
   self)
  (:description
   (&optional __description)
   (if __description (setq _description __description)) _description)
  (:result_code
   (&optional __result_code)
   (if __result_code (setq _result_code __result_code)) _result_code)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; string _description
    4 (length _description)
    ;; uint16 _result_code
    2
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _description
       (write-long (length _description) s) (princ _description s)
     ;; uint16 _result_code
       (write-word _result_code s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _description
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _description (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; uint16 _result_code
     (setq _result_code (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass abb_robot_msgs::GetRobotControllerDescription
  :super ros::object
  :slots ())

(setf (get abb_robot_msgs::GetRobotControllerDescription :md5sum-) "7c2a2ccf1dbdb360e4b26d80a3725b92")
(setf (get abb_robot_msgs::GetRobotControllerDescription :datatype-) "abb_robot_msgs/GetRobotControllerDescription")
(setf (get abb_robot_msgs::GetRobotControllerDescription :request) abb_robot_msgs::GetRobotControllerDescriptionRequest)
(setf (get abb_robot_msgs::GetRobotControllerDescription :response) abb_robot_msgs::GetRobotControllerDescriptionResponse)

(defmethod abb_robot_msgs::GetRobotControllerDescriptionRequest
  (:response () (instance abb_robot_msgs::GetRobotControllerDescriptionResponse :init)))

(setf (get abb_robot_msgs::GetRobotControllerDescriptionRequest :md5sum-) "7c2a2ccf1dbdb360e4b26d80a3725b92")
(setf (get abb_robot_msgs::GetRobotControllerDescriptionRequest :datatype-) "abb_robot_msgs/GetRobotControllerDescriptionRequest")
(setf (get abb_robot_msgs::GetRobotControllerDescriptionRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get a
#   (partial) description of an ABB robot controller system.
#
# Note:
#   The description will be returned in a Google Protocol Buffer text format.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Description of the robot controller system.
string description

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")

(setf (get abb_robot_msgs::GetRobotControllerDescriptionResponse :md5sum-) "7c2a2ccf1dbdb360e4b26d80a3725b92")
(setf (get abb_robot_msgs::GetRobotControllerDescriptionResponse :datatype-) "abb_robot_msgs/GetRobotControllerDescriptionResponse")
(setf (get abb_robot_msgs::GetRobotControllerDescriptionResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to get a
#   (partial) description of an ABB robot controller system.
#
# Note:
#   The description will be returned in a Google Protocol Buffer text format.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------

---

#-------------------------------------------------------------------------------
# Service response fields
#-------------------------------------------------------------------------------
# Description of the robot controller system.
string description

# Service success/failure indicator.
# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.
uint16 result_code

# Status message (empty if service succeeded).
string message

")



(provide :abb_robot_msgs/GetRobotControllerDescription "7c2a2ccf1dbdb360e4b26d80a3725b92")


