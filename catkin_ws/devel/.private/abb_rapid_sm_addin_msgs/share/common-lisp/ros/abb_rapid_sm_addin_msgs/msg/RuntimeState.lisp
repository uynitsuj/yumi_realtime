; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-msg)


;//! \htmlinclude RuntimeState.msg.html

(cl:defclass <RuntimeState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (state_machines
    :reader state_machines
    :initarg :state_machines
    :type (cl:vector abb_rapid_sm_addin_msgs-msg:StateMachineState)
   :initform (cl:make-array 0 :element-type 'abb_rapid_sm_addin_msgs-msg:StateMachineState :initial-element (cl:make-instance 'abb_rapid_sm_addin_msgs-msg:StateMachineState))))
)

(cl:defclass RuntimeState (<RuntimeState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RuntimeState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RuntimeState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-msg:<RuntimeState> is deprecated: use abb_rapid_sm_addin_msgs-msg:RuntimeState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <RuntimeState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:header-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'state_machines-val :lambda-list '(m))
(cl:defmethod state_machines-val ((m <RuntimeState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:state_machines-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:state_machines instead.")
  (state_machines m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RuntimeState>) ostream)
  "Serializes a message object of type '<RuntimeState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'state_machines))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'state_machines))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RuntimeState>) istream)
  "Deserializes a message object of type '<RuntimeState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'state_machines) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'state_machines)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'abb_rapid_sm_addin_msgs-msg:StateMachineState))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RuntimeState>)))
  "Returns string type for a message object of type '<RuntimeState>"
  "abb_rapid_sm_addin_msgs/RuntimeState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RuntimeState)))
  "Returns string type for a message object of type 'RuntimeState"
  "abb_rapid_sm_addin_msgs/RuntimeState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RuntimeState>)))
  "Returns md5sum for a message object of type '<RuntimeState>"
  "1f803b084f80e27cb1a45c04dc77c6e7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RuntimeState)))
  "Returns md5sum for a message object of type 'RuntimeState"
  "1f803b084f80e27cb1a45c04dc77c6e7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RuntimeState>)))
  "Returns full string definition for message of type '<RuntimeState>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the runtime states~%#   of all (known) RobotWare \"StateMachine Add-In\" RAPID instances.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Metadata.~%std_msgs/Header header~%~%# \"StateMachine Add-In\" instances.~%StateMachineState[] state_machines~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: abb_rapid_sm_addin_msgs/StateMachineState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the state of an~%#   instance of the RobotWare \"StateMachine Add-In\"'s RAPID implementation.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# Possible \"StateMachine Add-In\" RAPID states:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 SM_STATE_UNKNOWN           = 1~%uint8 SM_STATE_IDLE              = 2~%uint8 SM_STATE_INITIALIZE        = 3~%uint8 SM_STATE_RUN_RAPID_ROUTINE = 4~%uint8 SM_STATE_RUN_EGM_ROUTINE   = 5~%~%# Possible \"StateMachine Add-In\" RAPID EGM actions:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 EGM_ACTION_UNKNOWN      = 1~%uint8 EGM_ACTION_NONE         = 2~%uint8 EGM_ACTION_RUN_JOINT    = 3~%uint8 EGM_ACTION_RUN_POSE     = 4~%uint8 EGM_ACTION_STOP         = 5~%uint8 EGM_ACTION_START_STREAM = 6~%uint8 EGM_ACTION_STOP_STREAM  = 7~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string rapid_task~%~%# The \"StateMachine Add-In\" instance's current state.~%uint8 sm_state~%~%# The \"StateMachine Add-In\" instance's current EGM action~%# (only used if the RobotWare EGM option is present).~%uint8 egm_action~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RuntimeState)))
  "Returns full string definition for message of type 'RuntimeState"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the runtime states~%#   of all (known) RobotWare \"StateMachine Add-In\" RAPID instances.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Metadata.~%std_msgs/Header header~%~%# \"StateMachine Add-In\" instances.~%StateMachineState[] state_machines~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: abb_rapid_sm_addin_msgs/StateMachineState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the state of an~%#   instance of the RobotWare \"StateMachine Add-In\"'s RAPID implementation.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# Possible \"StateMachine Add-In\" RAPID states:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 SM_STATE_UNKNOWN           = 1~%uint8 SM_STATE_IDLE              = 2~%uint8 SM_STATE_INITIALIZE        = 3~%uint8 SM_STATE_RUN_RAPID_ROUTINE = 4~%uint8 SM_STATE_RUN_EGM_ROUTINE   = 5~%~%# Possible \"StateMachine Add-In\" RAPID EGM actions:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 EGM_ACTION_UNKNOWN      = 1~%uint8 EGM_ACTION_NONE         = 2~%uint8 EGM_ACTION_RUN_JOINT    = 3~%uint8 EGM_ACTION_RUN_POSE     = 4~%uint8 EGM_ACTION_STOP         = 5~%uint8 EGM_ACTION_START_STREAM = 6~%uint8 EGM_ACTION_STOP_STREAM  = 7~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string rapid_task~%~%# The \"StateMachine Add-In\" instance's current state.~%uint8 sm_state~%~%# The \"StateMachine Add-In\" instance's current EGM action~%# (only used if the RobotWare EGM option is present).~%uint8 egm_action~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RuntimeState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'state_machines) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RuntimeState>))
  "Converts a ROS message object to a list"
  (cl:list 'RuntimeState
    (cl:cons ':header (header msg))
    (cl:cons ':state_machines (state_machines msg))
))
