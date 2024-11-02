; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-msg)


;//! \htmlinclude StateMachineState.msg.html

(cl:defclass <StateMachineState> (roslisp-msg-protocol:ros-message)
  ((rapid_task
    :reader rapid_task
    :initarg :rapid_task
    :type cl:string
    :initform "")
   (sm_state
    :reader sm_state
    :initarg :sm_state
    :type cl:fixnum
    :initform 0)
   (egm_action
    :reader egm_action
    :initarg :egm_action
    :type cl:fixnum
    :initform 0))
)

(cl:defclass StateMachineState (<StateMachineState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <StateMachineState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'StateMachineState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-msg:<StateMachineState> is deprecated: use abb_rapid_sm_addin_msgs-msg:StateMachineState instead.")))

(cl:ensure-generic-function 'rapid_task-val :lambda-list '(m))
(cl:defmethod rapid_task-val ((m <StateMachineState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:rapid_task-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:rapid_task instead.")
  (rapid_task m))

(cl:ensure-generic-function 'sm_state-val :lambda-list '(m))
(cl:defmethod sm_state-val ((m <StateMachineState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:sm_state-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:sm_state instead.")
  (sm_state m))

(cl:ensure-generic-function 'egm_action-val :lambda-list '(m))
(cl:defmethod egm_action-val ((m <StateMachineState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-msg:egm_action-val is deprecated.  Use abb_rapid_sm_addin_msgs-msg:egm_action instead.")
  (egm_action m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<StateMachineState>)))
    "Constants for message type '<StateMachineState>"
  '((:SM_STATE_UNKNOWN . 1)
    (:SM_STATE_IDLE . 2)
    (:SM_STATE_INITIALIZE . 3)
    (:SM_STATE_RUN_RAPID_ROUTINE . 4)
    (:SM_STATE_RUN_EGM_ROUTINE . 5)
    (:EGM_ACTION_UNKNOWN . 1)
    (:EGM_ACTION_NONE . 2)
    (:EGM_ACTION_RUN_JOINT . 3)
    (:EGM_ACTION_RUN_POSE . 4)
    (:EGM_ACTION_STOP . 5)
    (:EGM_ACTION_START_STREAM . 6)
    (:EGM_ACTION_STOP_STREAM . 7))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'StateMachineState)))
    "Constants for message type 'StateMachineState"
  '((:SM_STATE_UNKNOWN . 1)
    (:SM_STATE_IDLE . 2)
    (:SM_STATE_INITIALIZE . 3)
    (:SM_STATE_RUN_RAPID_ROUTINE . 4)
    (:SM_STATE_RUN_EGM_ROUTINE . 5)
    (:EGM_ACTION_UNKNOWN . 1)
    (:EGM_ACTION_NONE . 2)
    (:EGM_ACTION_RUN_JOINT . 3)
    (:EGM_ACTION_RUN_POSE . 4)
    (:EGM_ACTION_STOP . 5)
    (:EGM_ACTION_START_STREAM . 6)
    (:EGM_ACTION_STOP_STREAM . 7))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <StateMachineState>) ostream)
  "Serializes a message object of type '<StateMachineState>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'rapid_task))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'rapid_task))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sm_state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'egm_action)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <StateMachineState>) istream)
  "Deserializes a message object of type '<StateMachineState>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'rapid_task) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'rapid_task) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sm_state)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'egm_action)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<StateMachineState>)))
  "Returns string type for a message object of type '<StateMachineState>"
  "abb_rapid_sm_addin_msgs/StateMachineState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'StateMachineState)))
  "Returns string type for a message object of type 'StateMachineState"
  "abb_rapid_sm_addin_msgs/StateMachineState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<StateMachineState>)))
  "Returns md5sum for a message object of type '<StateMachineState>"
  "de6354e383149183de9c15ab3d871c7b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'StateMachineState)))
  "Returns md5sum for a message object of type 'StateMachineState"
  "de6354e383149183de9c15ab3d871c7b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<StateMachineState>)))
  "Returns full string definition for message of type '<StateMachineState>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the state of an~%#   instance of the RobotWare \"StateMachine Add-In\"'s RAPID implementation.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# Possible \"StateMachine Add-In\" RAPID states:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 SM_STATE_UNKNOWN           = 1~%uint8 SM_STATE_IDLE              = 2~%uint8 SM_STATE_INITIALIZE        = 3~%uint8 SM_STATE_RUN_RAPID_ROUTINE = 4~%uint8 SM_STATE_RUN_EGM_ROUTINE   = 5~%~%# Possible \"StateMachine Add-In\" RAPID EGM actions:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 EGM_ACTION_UNKNOWN      = 1~%uint8 EGM_ACTION_NONE         = 2~%uint8 EGM_ACTION_RUN_JOINT    = 3~%uint8 EGM_ACTION_RUN_POSE     = 4~%uint8 EGM_ACTION_STOP         = 5~%uint8 EGM_ACTION_START_STREAM = 6~%uint8 EGM_ACTION_STOP_STREAM  = 7~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string rapid_task~%~%# The \"StateMachine Add-In\" instance's current state.~%uint8 sm_state~%~%# The \"StateMachine Add-In\" instance's current EGM action~%# (only used if the RobotWare EGM option is present).~%uint8 egm_action~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'StateMachineState)))
  "Returns full string definition for message of type 'StateMachineState"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the state of an~%#   instance of the RobotWare \"StateMachine Add-In\"'s RAPID implementation.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# Possible \"StateMachine Add-In\" RAPID states:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 SM_STATE_UNKNOWN           = 1~%uint8 SM_STATE_IDLE              = 2~%uint8 SM_STATE_INITIALIZE        = 3~%uint8 SM_STATE_RUN_RAPID_ROUTINE = 4~%uint8 SM_STATE_RUN_EGM_ROUTINE   = 5~%~%# Possible \"StateMachine Add-In\" RAPID EGM actions:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 EGM_ACTION_UNKNOWN      = 1~%uint8 EGM_ACTION_NONE         = 2~%uint8 EGM_ACTION_RUN_JOINT    = 3~%uint8 EGM_ACTION_RUN_POSE     = 4~%uint8 EGM_ACTION_STOP         = 5~%uint8 EGM_ACTION_START_STREAM = 6~%uint8 EGM_ACTION_STOP_STREAM  = 7~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string rapid_task~%~%# The \"StateMachine Add-In\" instance's current state.~%uint8 sm_state~%~%# The \"StateMachine Add-In\" instance's current EGM action~%# (only used if the RobotWare EGM option is present).~%uint8 egm_action~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <StateMachineState>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'rapid_task))
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <StateMachineState>))
  "Converts a ROS message object to a list"
  (cl:list 'StateMachineState
    (cl:cons ':rapid_task (rapid_task msg))
    (cl:cons ':sm_state (sm_state msg))
    (cl:cons ':egm_action (egm_action msg))
))
