;; Auto-generated. Do not edit!


(when (boundp 'abb_rapid_sm_addin_msgs::RuntimeState)
  (if (not (find-package "ABB_RAPID_SM_ADDIN_MSGS"))
    (make-package "ABB_RAPID_SM_ADDIN_MSGS"))
  (shadow 'RuntimeState (find-package "ABB_RAPID_SM_ADDIN_MSGS")))
(unless (find-package "ABB_RAPID_SM_ADDIN_MSGS::RUNTIMESTATE")
  (make-package "ABB_RAPID_SM_ADDIN_MSGS::RUNTIMESTATE"))

(in-package "ROS")
;;//! \htmlinclude RuntimeState.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass abb_rapid_sm_addin_msgs::RuntimeState
  :super ros::object
  :slots (_header _state_machines ))

(defmethod abb_rapid_sm_addin_msgs::RuntimeState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:state_machines __state_machines) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _state_machines __state_machines)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:state_machines
   (&rest __state_machines)
   (if (keywordp (car __state_machines))
       (send* _state_machines __state_machines)
     (progn
       (if __state_machines (setq _state_machines (car __state_machines)))
       _state_machines)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; abb_rapid_sm_addin_msgs/StateMachineState[] _state_machines
    (apply #'+ (send-all _state_machines :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; abb_rapid_sm_addin_msgs/StateMachineState[] _state_machines
     (write-long (length _state_machines) s)
     (dolist (elem _state_machines)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; abb_rapid_sm_addin_msgs/StateMachineState[] _state_machines
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _state_machines (let (r) (dotimes (i n) (push (instance abb_rapid_sm_addin_msgs::StateMachineState :init) r)) r))
     (dolist (elem- _state_machines)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get abb_rapid_sm_addin_msgs::RuntimeState :md5sum-) "1f803b084f80e27cb1a45c04dc77c6e7")
(setf (get abb_rapid_sm_addin_msgs::RuntimeState :datatype-) "abb_rapid_sm_addin_msgs/RuntimeState")
(setf (get abb_rapid_sm_addin_msgs::RuntimeState :definition-)
      "#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the runtime states
#   of all (known) RobotWare \"StateMachine Add-In\" RAPID instances.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# Metadata.
std_msgs/Header header

# \"StateMachine Add-In\" instances.
StateMachineState[] state_machines

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: abb_rapid_sm_addin_msgs/StateMachineState
#-------------------------------------------------------------------------------
# Description:
#   The purpose of this message definition, is to represent the state of an
#   instance of the RobotWare \"StateMachine Add-In\"'s RAPID implementation.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Enumerations
#-------------------------------------------------------------------------------
# Possible \"StateMachine Add-In\" RAPID states:
#
# Note: These enumeration values differ from the RAPID implementation
#       values to avoid conflict with default values in ROS messages.
uint8 SM_STATE_UNKNOWN           = 1
uint8 SM_STATE_IDLE              = 2
uint8 SM_STATE_INITIALIZE        = 3
uint8 SM_STATE_RUN_RAPID_ROUTINE = 4
uint8 SM_STATE_RUN_EGM_ROUTINE   = 5

# Possible \"StateMachine Add-In\" RAPID EGM actions:
#
# Note: These enumeration values differ from the RAPID implementation
#       values to avoid conflict with default values in ROS messages.
uint8 EGM_ACTION_UNKNOWN      = 1
uint8 EGM_ACTION_NONE         = 2
uint8 EGM_ACTION_RUN_JOINT    = 3
uint8 EGM_ACTION_RUN_POSE     = 4
uint8 EGM_ACTION_STOP         = 5
uint8 EGM_ACTION_START_STREAM = 6
uint8 EGM_ACTION_STOP_STREAM  = 7

#-------------------------------------------------------------------------------
# Message fields
#-------------------------------------------------------------------------------
# The RAPID task that runs the \"StateMachine Add-In\" instance.
string rapid_task

# The \"StateMachine Add-In\" instance's current state.
uint8 sm_state

# The \"StateMachine Add-In\" instance's current EGM action
# (only used if the RobotWare EGM option is present).
uint8 egm_action

")



(provide :abb_rapid_sm_addin_msgs/RuntimeState "1f803b084f80e27cb1a45c04dc77c6e7")


