;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::SetSGCommand)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'SetSGCommand (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMAND")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMAND"))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDRESPONSE")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDRESPONSE"))

(in-package "ROS")





(intern "*SG_COMMAND_UNKNOWN*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_UNKNOWN* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_UNKNOWN* 1)
(intern "*SG_COMMAND_NONE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_NONE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_NONE* 2)
(intern "*SG_COMMAND_INITIALIZE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_INITIALIZE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_INITIALIZE* 3)
(intern "*SG_COMMAND_CALIBRATE*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_CALIBRATE* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_CALIBRATE* 4)
(intern "*SG_COMMAND_MOVE_TO*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_MOVE_TO* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_MOVE_TO* 5)
(intern "*SG_COMMAND_GRIP_IN*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_GRIP_IN* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_GRIP_IN* 6)
(intern "*SG_COMMAND_GRIP_OUT*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_GRIP_OUT* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_GRIP_OUT* 7)
(intern "*SG_COMMAND_BLOW_ON_1*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_BLOW_ON_1* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_ON_1* 8)
(intern "*SG_COMMAND_BLOW_ON_2*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_BLOW_ON_2* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_ON_2* 9)
(intern "*SG_COMMAND_BLOW_OFF_1*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_BLOW_OFF_1* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_OFF_1* 10)
(intern "*SG_COMMAND_BLOW_OFF_2*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_BLOW_OFF_2* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_OFF_2* 11)
(intern "*SG_COMMAND_VACUUM_ON_1*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_VACUUM_ON_1* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_ON_1* 12)
(intern "*SG_COMMAND_VACUUM_ON_2*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_VACUUM_ON_2* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_ON_2* 13)
(intern "*SG_COMMAND_VACUUM_OFF_1*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_VACUUM_OFF_1* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_OFF_1* 14)
(intern "*SG_COMMAND_VACUUM_OFF_2*" (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(shadow '*SG_COMMAND_VACUUM_OFF_2* (find-package "ABB_RAPID_SM_ADDIN_MSGS::SETSGCOMMANDREQUEST"))
(defconstant abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_OFF_2* 15)

(defun abb_rapid_sm_addin_msgs::SetSGCommandRequest-to-symbol (const)
  (cond
        ((= const 1) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_UNKNOWN*)
        ((= const 2) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_NONE*)
        ((= const 3) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_INITIALIZE*)
        ((= const 4) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_CALIBRATE*)
        ((= const 5) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_MOVE_TO*)
        ((= const 6) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_GRIP_IN*)
        ((= const 7) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_GRIP_OUT*)
        ((= const 8) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_ON_1*)
        ((= const 9) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_ON_2*)
        ((= const 10) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_OFF_1*)
        ((= const 11) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_BLOW_OFF_2*)
        ((= const 12) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_ON_1*)
        ((= const 13) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_ON_2*)
        ((= const 14) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_OFF_1*)
        ((= const 15) 'abb_rapid_sm_addin_msgs::SetSGCommandRequest::*SG_COMMAND_VACUUM_OFF_2*)
        (t nil)))

(defclass abb_rapid_sm_addin_msgs::SetSGCommandRequest
  :super ros::object
  :slots (_task _command _target_position ))

(defmethod abb_rapid_sm_addin_msgs::SetSGCommandRequest
  (:init
   (&key
    ((:task __task) "")
    ((:command __command) 0)
    ((:target_position __target_position) 0.0)
    )
   (send-super :init)
   (setq _task (string __task))
   (setq _command (round __command))
   (setq _target_position (float __target_position))
   self)
  (:task
   (&optional __task)
   (if __task (setq _task __task)) _task)
  (:command
   (&optional __command)
   (if __command (setq _command __command)) _command)
  (:target_position
   (&optional __target_position)
   (if __target_position (setq _target_position __target_position)) _target_position)
  (:serialization-length
   ()
   (+
    ;; string _task
    4 (length _task)
    ;; uint8 _command
    1
    ;; float32 _target_position
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _task
       (write-long (length _task) s) (princ _task s)
     ;; uint8 _command
       (write-byte _command s)
     ;; float32 _target_position
       (sys::poke _target_position (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _task
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _task (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; uint8 _command
     (setq _command (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32 _target_position
     (setq _target_position (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass abb_rapid_sm_addin_msgs::SetSGCommandResponse
  :super ros::object
  :slots (_result_code _message ))

(defmethod abb_rapid_sm_addin_msgs::SetSGCommandResponse
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

(defclass abb_rapid_sm_addin_msgs::SetSGCommand
  :super ros::object
  :slots ())

(setf (get abb_rapid_sm_addin_msgs::SetSGCommand :md5sum-) "7aa352af5c8c7b889375c50673d12253")
(setf (get abb_rapid_sm_addin_msgs::SetSGCommand :datatype-) "abb_rapid_sm_addin_msgs/SetSGCommand")
(setf (get abb_rapid_sm_addin_msgs::SetSGCommand :request) abb_rapid_sm_addin_msgs::SetSGCommandRequest)
(setf (get abb_rapid_sm_addin_msgs::SetSGCommand :response) abb_rapid_sm_addin_msgs::SetSGCommandResponse)

(defmethod abb_rapid_sm_addin_msgs::SetSGCommandRequest
  (:response () (instance abb_rapid_sm_addin_msgs::SetSGCommandResponse :init)))

(setf (get abb_rapid_sm_addin_msgs::SetSGCommandRequest :md5sum-) "7aa352af5c8c7b889375c50673d12253")
(setf (get abb_rapid_sm_addin_msgs::SetSGCommandRequest :datatype-) "abb_rapid_sm_addin_msgs/SetSGCommandRequest")
(setf (get abb_rapid_sm_addin_msgs::SetSGCommandRequest :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the values
#   of predefined RAPID variables for commanding a SmartGripper tool.
#
# Note:
#   These variables are defined in each RAPID motion task, in ABB robot
#   controller systems, that are using the RobotWare \"StateMachine Add-In\"
#   together with a SmartGripper tool.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# Supported \"StateMachine Add-In\" SmartGripper commands:
#
# Note: These enumeration values differ from the RAPID implementation
#       values to avoid conflict with default values in ROS messages.
uint8 SG_COMMAND_UNKNOWN      = 1
uint8 SG_COMMAND_NONE         = 2
uint8 SG_COMMAND_INITIALIZE   = 3
uint8 SG_COMMAND_CALIBRATE    = 4
uint8 SG_COMMAND_MOVE_TO      = 5
uint8 SG_COMMAND_GRIP_IN      = 6
uint8 SG_COMMAND_GRIP_OUT     = 7
uint8 SG_COMMAND_BLOW_ON_1    = 8
uint8 SG_COMMAND_BLOW_ON_2    = 9
uint8 SG_COMMAND_BLOW_OFF_1   = 10
uint8 SG_COMMAND_BLOW_OFF_2   = 11
uint8 SG_COMMAND_VACUUM_ON_1  = 12
uint8 SG_COMMAND_VACUUM_ON_2  = 13
uint8 SG_COMMAND_VACUUM_OFF_1 = 14
uint8 SG_COMMAND_VACUUM_OFF_2 = 15

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# The RAPID task that runs the \"StateMachine Add-In\" instance.
string task

# Desired SmartGripper command.
uint8 command

# Desired target position [mm] (only used for the 'SG_COMMAND_MOVE_TO' command).
float32 target_position

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

(setf (get abb_rapid_sm_addin_msgs::SetSGCommandResponse :md5sum-) "7aa352af5c8c7b889375c50673d12253")
(setf (get abb_rapid_sm_addin_msgs::SetSGCommandResponse :datatype-) "abb_rapid_sm_addin_msgs/SetSGCommandResponse")
(setf (get abb_rapid_sm_addin_msgs::SetSGCommandResponse :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this service definition, is to define a way to set the values
#   of predefined RAPID variables for commanding a SmartGripper tool.
#
# Note:
#   These variables are defined in each RAPID motion task, in ABB robot
#   controller systems, that are using the RobotWare \"StateMachine Add-In\"
#   together with a SmartGripper tool.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# Supported \"StateMachine Add-In\" SmartGripper commands:
#
# Note: These enumeration values differ from the RAPID implementation
#       values to avoid conflict with default values in ROS messages.
uint8 SG_COMMAND_UNKNOWN      = 1
uint8 SG_COMMAND_NONE         = 2
uint8 SG_COMMAND_INITIALIZE   = 3
uint8 SG_COMMAND_CALIBRATE    = 4
uint8 SG_COMMAND_MOVE_TO      = 5
uint8 SG_COMMAND_GRIP_IN      = 6
uint8 SG_COMMAND_GRIP_OUT     = 7
uint8 SG_COMMAND_BLOW_ON_1    = 8
uint8 SG_COMMAND_BLOW_ON_2    = 9
uint8 SG_COMMAND_BLOW_OFF_1   = 10
uint8 SG_COMMAND_BLOW_OFF_2   = 11
uint8 SG_COMMAND_VACUUM_ON_1  = 12
uint8 SG_COMMAND_VACUUM_ON_2  = 13
uint8 SG_COMMAND_VACUUM_OFF_1 = 14
uint8 SG_COMMAND_VACUUM_OFF_2 = 15

#-------------------------------------------------------------------------------
# Service request fields
#-------------------------------------------------------------------------------
# The RAPID task that runs the \"StateMachine Add-In\" instance.
string task

# Desired SmartGripper command.
uint8 command

# Desired target position [mm] (only used for the 'SG_COMMAND_MOVE_TO' command).
float32 target_position

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



(provide :abb_rapid_sm_addin_msgs/SetSGCommand "7aa352af5c8c7b889375c50673d12253")


