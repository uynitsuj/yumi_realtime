;; Auto-generated. Do not edit!


(when (boundp 'abb_robot_msgs::ServiceResponses)
  (if (not (find-package "ABB_ROBOT_MSGS"))
    (make-package "ABB_ROBOT_MSGS"))
  (shadow 'ServiceResponses (find-package "ABB_ROBOT_MSGS")))
(unless (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES")
  (make-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))

(in-package "ROS")
;;//! \htmlinclude ServiceResponses.msg.html


(intern "*SUCCESS*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*SUCCESS* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*SUCCESS* "\"\"")
(intern "*RC_SUCCESS*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_SUCCESS* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_SUCCESS* 1)
(intern "*FAILED*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*FAILED* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*FAILED* "\"Service failed (e.g. timeout or resource not found, see the DEBUG log for details)\"")
(intern "*RC_FAILED*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_FAILED* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_FAILED* 2)
(intern "*SERVER_IS_BUSY*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*SERVER_IS_BUSY* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*SERVER_IS_BUSY* "\"Server is busy\"")
(intern "*RC_SERVER_IS_BUSY*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_SERVER_IS_BUSY* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_SERVER_IS_BUSY* 1001)
(intern "*EMPTY_FILENAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*EMPTY_FILENAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*EMPTY_FILENAME* "\"Filename is empty\"")
(intern "*RC_EMPTY_FILENAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_EMPTY_FILENAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_EMPTY_FILENAME* 2001)
(intern "*EMPTY_SIGNAL_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*EMPTY_SIGNAL_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*EMPTY_SIGNAL_NAME* "\"Signal name is empty\"")
(intern "*RC_EMPTY_SIGNAL_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_EMPTY_SIGNAL_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_EMPTY_SIGNAL_NAME* 2002)
(intern "*EMPTY_RAPID_TASK_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*EMPTY_RAPID_TASK_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*EMPTY_RAPID_TASK_NAME* "\"RAPID task name is empty\"")
(intern "*RC_EMPTY_RAPID_TASK_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_EMPTY_RAPID_TASK_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_EMPTY_RAPID_TASK_NAME* 2003)
(intern "*EMPTY_RAPID_MODULE_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*EMPTY_RAPID_MODULE_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*EMPTY_RAPID_MODULE_NAME* "\"RAPID module name is empty\"")
(intern "*RC_EMPTY_RAPID_MODULE_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_EMPTY_RAPID_MODULE_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_EMPTY_RAPID_MODULE_NAME* 2004)
(intern "*EMPTY_RAPID_SYMBOL_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*EMPTY_RAPID_SYMBOL_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*EMPTY_RAPID_SYMBOL_NAME* "\"RAPID symbol name is empty\"")
(intern "*RC_EMPTY_RAPID_SYMBOL_NAME*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_EMPTY_RAPID_SYMBOL_NAME* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_EMPTY_RAPID_SYMBOL_NAME* 2005)
(intern "*NOT_IN_AUTO_MODE*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*NOT_IN_AUTO_MODE* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*NOT_IN_AUTO_MODE* "\"Robot controller is not in AUTO mode\"")
(intern "*RC_NOT_IN_AUTO_MODE*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_NOT_IN_AUTO_MODE* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_NOT_IN_AUTO_MODE* 3001)
(intern "*MOTORS_ARE_OFF*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*MOTORS_ARE_OFF* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*MOTORS_ARE_OFF* "\"Motors are off\"")
(intern "*RC_MOTORS_ARE_OFF*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_MOTORS_ARE_OFF* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_MOTORS_ARE_OFF* 3002)
(intern "*MOTORS_ARE_ON*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*MOTORS_ARE_ON* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*MOTORS_ARE_ON* "\"Motors are on\"")
(intern "*RC_MOTORS_ARE_ON*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_MOTORS_ARE_ON* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_MOTORS_ARE_ON* 3003)
(intern "*RAPID_NOT_STOPPED*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RAPID_NOT_STOPPED* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RAPID_NOT_STOPPED* "\"RAPID has not been stopped\"")
(intern "*RC_RAPID_NOT_STOPPED*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_RAPID_NOT_STOPPED* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_RAPID_NOT_STOPPED* 3004)
(intern "*RAPID_NOT_RUNNING*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RAPID_NOT_RUNNING* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RAPID_NOT_RUNNING* "\"RAPID is not running\"")
(intern "*RC_RAPID_NOT_RUNNING*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_RAPID_NOT_RUNNING* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_RAPID_NOT_RUNNING* 3005)
(intern "*SM_RUNTIME_STATES_MISSING*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*SM_RUNTIME_STATES_MISSING* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*SM_RUNTIME_STATES_MISSING* "\"No runtime states received for any StateMachine Add-In instance\"")
(intern "*RC_SM_RUNTIME_STATES_MISSING*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_SM_RUNTIME_STATES_MISSING* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_SM_RUNTIME_STATES_MISSING* 4001)
(intern "*SM_UNKNOWN_RAPID_TASK*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*SM_UNKNOWN_RAPID_TASK* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*SM_UNKNOWN_RAPID_TASK* "\"RAPID task not found among known StateMachine Add-In instances\"")
(intern "*RC_SM_UNKNOWN_RAPID_TASK*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_SM_UNKNOWN_RAPID_TASK* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_SM_UNKNOWN_RAPID_TASK* 4002)
(intern "*SM_UNINITIALIZED*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*SM_UNINITIALIZED* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*SM_UNINITIALIZED* "\"StateMachine Add-In instance has not been initialized\"")
(intern "*RC_SM_UNINITIALIZED*" (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(shadow '*RC_SM_UNINITIALIZED* (find-package "ABB_ROBOT_MSGS::SERVICERESPONSES"))
(defconstant abb_robot_msgs::ServiceResponses::*RC_SM_UNINITIALIZED* 4003)

(defun abb_robot_msgs::ServiceResponses-to-symbol (const)
  (cond
        ((= const 1) 'abb_robot_msgs::ServiceResponses::*RC_SUCCESS*)
        ((= const 2) 'abb_robot_msgs::ServiceResponses::*RC_FAILED*)
        ((= const 1001) 'abb_robot_msgs::ServiceResponses::*RC_SERVER_IS_BUSY*)
        ((= const 2001) 'abb_robot_msgs::ServiceResponses::*RC_EMPTY_FILENAME*)
        ((= const 2002) 'abb_robot_msgs::ServiceResponses::*RC_EMPTY_SIGNAL_NAME*)
        ((= const 2003) 'abb_robot_msgs::ServiceResponses::*RC_EMPTY_RAPID_TASK_NAME*)
        ((= const 2004) 'abb_robot_msgs::ServiceResponses::*RC_EMPTY_RAPID_MODULE_NAME*)
        ((= const 2005) 'abb_robot_msgs::ServiceResponses::*RC_EMPTY_RAPID_SYMBOL_NAME*)
        ((= const 3001) 'abb_robot_msgs::ServiceResponses::*RC_NOT_IN_AUTO_MODE*)
        ((= const 3002) 'abb_robot_msgs::ServiceResponses::*RC_MOTORS_ARE_OFF*)
        ((= const 3003) 'abb_robot_msgs::ServiceResponses::*RC_MOTORS_ARE_ON*)
        ((= const 3004) 'abb_robot_msgs::ServiceResponses::*RC_RAPID_NOT_STOPPED*)
        ((= const 3005) 'abb_robot_msgs::ServiceResponses::*RC_RAPID_NOT_RUNNING*)
        ((= const 4001) 'abb_robot_msgs::ServiceResponses::*RC_SM_RUNTIME_STATES_MISSING*)
        ((= const 4002) 'abb_robot_msgs::ServiceResponses::*RC_SM_UNKNOWN_RAPID_TASK*)
        ((= const 4003) 'abb_robot_msgs::ServiceResponses::*RC_SM_UNINITIALIZED*)
        (t nil)))

(defclass abb_robot_msgs::ServiceResponses
  :super ros::object
  :slots ())

(defmethod abb_robot_msgs::ServiceResponses
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

(setf (get abb_robot_msgs::ServiceResponses :md5sum-) "c541477c2ff72df8851a0d0cd0b0b782")
(setf (get abb_robot_msgs::ServiceResponses :datatype-) "abb_robot_msgs/ServiceResponses")
(setf (get abb_robot_msgs::ServiceResponses :definition-)
      "# result codes are 16 bit unsigned

# Generic OK
string SUCCESS=\"\"
uint16 RC_SUCCESS=1

# Generic failure message
string FAILED=\"Service failed (e.g. timeout or resource not found, see the DEBUG log for details)\"
uint16 RC_FAILED=2

# The server's underlying communication interface is busy
string SERVER_IS_BUSY=\"Server is busy\"
uint16 RC_SERVER_IS_BUSY=1001

# Filename field is empty in service request
string EMPTY_FILENAME=\"Filename is empty\"
uint16 RC_EMPTY_FILENAME=2001

# Signal name field is empty in service request
string EMPTY_SIGNAL_NAME=\"Signal name is empty\"
uint16 RC_EMPTY_SIGNAL_NAME=2002

# RAPID task name field is empty in service request
string EMPTY_RAPID_TASK_NAME=\"RAPID task name is empty\"
uint16 RC_EMPTY_RAPID_TASK_NAME=2003

# RAPID module name field is empty in service request
string EMPTY_RAPID_MODULE_NAME=\"RAPID module name is empty\"
uint16 RC_EMPTY_RAPID_MODULE_NAME=2004

# RAPID symbol name field is empty in service request
string EMPTY_RAPID_SYMBOL_NAME=\"RAPID symbol name is empty\"
uint16 RC_EMPTY_RAPID_SYMBOL_NAME=2005

# Robot controller is not in auto mode
string NOT_IN_AUTO_MODE=\"Robot controller is not in AUTO mode\"
uint16 RC_NOT_IN_AUTO_MODE=3001

# The motors are off
string MOTORS_ARE_OFF=\"Motors are off\"
uint16 RC_MOTORS_ARE_OFF=3002

# The motors are on
string MOTORS_ARE_ON=\"Motors are on\"
uint16 RC_MOTORS_ARE_ON=3003

# RAPID has not been stopped
string RAPID_NOT_STOPPED=\"RAPID has not been stopped\"
uint16 RC_RAPID_NOT_STOPPED=3004

# RAPID is not running
string RAPID_NOT_RUNNING=\"RAPID is not running\"
uint16 RC_RAPID_NOT_RUNNING=3005

# RobotWare StateMachine Add-In runtime state is missing
string SM_RUNTIME_STATES_MISSING=\"No runtime states received for any StateMachine Add-In instance\"
uint16 RC_SM_RUNTIME_STATES_MISSING=4001

# RobotWare StateMachine Add-In-instances do not know RAPID task
string SM_UNKNOWN_RAPID_TASK=\"RAPID task not found among known StateMachine Add-In instances\"
uint16 RC_SM_UNKNOWN_RAPID_TASK=4002

# RobotWare StateMachine Add-In instance has not been initialized
string SM_UNINITIALIZED=\"StateMachine Add-In instance has not been initialized\"
uint16 RC_SM_UNINITIALIZED=4003

")



(provide :abb_robot_msgs/ServiceResponses "c541477c2ff72df8851a0d0cd0b0b782")


