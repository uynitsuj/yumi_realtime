; Auto-generated. Do not edit!


(cl:in-package abb_egm_msgs-msg)


;//! \htmlinclude EGMChannelState.msg.html

(cl:defclass <EGMChannelState> (roslisp-msg-protocol:ros-message)
  ((name
    :reader name
    :initarg :name
    :type cl:string
    :initform "")
   (active
    :reader active
    :initarg :active
    :type cl:boolean
    :initform cl:nil)
   (egm_convergence_met
    :reader egm_convergence_met
    :initarg :egm_convergence_met
    :type cl:boolean
    :initform cl:nil)
   (egm_client_state
    :reader egm_client_state
    :initarg :egm_client_state
    :type cl:fixnum
    :initform 0)
   (motor_state
    :reader motor_state
    :initarg :motor_state
    :type cl:fixnum
    :initform 0)
   (rapid_execution_state
    :reader rapid_execution_state
    :initarg :rapid_execution_state
    :type cl:fixnum
    :initform 0)
   (utilization_rate
    :reader utilization_rate
    :initarg :utilization_rate
    :type cl:float
    :initform 0.0))
)

(cl:defclass EGMChannelState (<EGMChannelState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EGMChannelState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EGMChannelState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_egm_msgs-msg:<EGMChannelState> is deprecated: use abb_egm_msgs-msg:EGMChannelState instead.")))

(cl:ensure-generic-function 'name-val :lambda-list '(m))
(cl:defmethod name-val ((m <EGMChannelState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:name-val is deprecated.  Use abb_egm_msgs-msg:name instead.")
  (name m))

(cl:ensure-generic-function 'active-val :lambda-list '(m))
(cl:defmethod active-val ((m <EGMChannelState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:active-val is deprecated.  Use abb_egm_msgs-msg:active instead.")
  (active m))

(cl:ensure-generic-function 'egm_convergence_met-val :lambda-list '(m))
(cl:defmethod egm_convergence_met-val ((m <EGMChannelState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:egm_convergence_met-val is deprecated.  Use abb_egm_msgs-msg:egm_convergence_met instead.")
  (egm_convergence_met m))

(cl:ensure-generic-function 'egm_client_state-val :lambda-list '(m))
(cl:defmethod egm_client_state-val ((m <EGMChannelState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:egm_client_state-val is deprecated.  Use abb_egm_msgs-msg:egm_client_state instead.")
  (egm_client_state m))

(cl:ensure-generic-function 'motor_state-val :lambda-list '(m))
(cl:defmethod motor_state-val ((m <EGMChannelState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:motor_state-val is deprecated.  Use abb_egm_msgs-msg:motor_state instead.")
  (motor_state m))

(cl:ensure-generic-function 'rapid_execution_state-val :lambda-list '(m))
(cl:defmethod rapid_execution_state-val ((m <EGMChannelState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:rapid_execution_state-val is deprecated.  Use abb_egm_msgs-msg:rapid_execution_state instead.")
  (rapid_execution_state m))

(cl:ensure-generic-function 'utilization_rate-val :lambda-list '(m))
(cl:defmethod utilization_rate-val ((m <EGMChannelState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:utilization_rate-val is deprecated.  Use abb_egm_msgs-msg:utilization_rate instead.")
  (utilization_rate m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<EGMChannelState>)))
    "Constants for message type '<EGMChannelState>"
  '((:EGM_UNDEFINED . 1)
    (:EGM_ERROR . 2)
    (:EGM_STOPPED . 3)
    (:EGM_RUNNING . 4)
    (:MOTORS_UNDEFINED . 1)
    (:MOTORS_ON . 2)
    (:MOTORS_OFF . 3)
    (:RAPID_UNDEFINED . 1)
    (:RAPID_STOPPED . 2)
    (:RAPID_RUNNING . 3))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'EGMChannelState)))
    "Constants for message type 'EGMChannelState"
  '((:EGM_UNDEFINED . 1)
    (:EGM_ERROR . 2)
    (:EGM_STOPPED . 3)
    (:EGM_RUNNING . 4)
    (:MOTORS_UNDEFINED . 1)
    (:MOTORS_ON . 2)
    (:MOTORS_OFF . 3)
    (:RAPID_UNDEFINED . 1)
    (:RAPID_STOPPED . 2)
    (:RAPID_RUNNING . 3))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EGMChannelState>) ostream)
  "Serializes a message object of type '<EGMChannelState>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'name))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'active) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'egm_convergence_met) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'egm_client_state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'motor_state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'rapid_execution_state)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'utilization_rate))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EGMChannelState>) istream)
  "Deserializes a message object of type '<EGMChannelState>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'active) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'egm_convergence_met) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'egm_client_state)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'motor_state)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'rapid_execution_state)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'utilization_rate) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EGMChannelState>)))
  "Returns string type for a message object of type '<EGMChannelState>"
  "abb_egm_msgs/EGMChannelState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EGMChannelState)))
  "Returns string type for a message object of type 'EGMChannelState"
  "abb_egm_msgs/EGMChannelState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EGMChannelState>)))
  "Returns md5sum for a message object of type '<EGMChannelState>"
  "bc28749df58cd58b7299c4217fd9899f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EGMChannelState)))
  "Returns md5sum for a message object of type 'EGMChannelState"
  "bc28749df58cd58b7299c4217fd9899f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EGMChannelState>)))
  "Returns full string definition for message of type '<EGMChannelState>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of an EGM communication channel. I.e. a channel between an EGM client,~%#   in an ABB robot controller system, and an external EGM server.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# EGM client states:~%uint8 EGM_UNDEFINED = 1~%uint8 EGM_ERROR     = 2~%uint8 EGM_STOPPED   = 3~%uint8 EGM_RUNNING   = 4~%~%# Motor states:~%uint8 MOTORS_UNDEFINED = 1~%uint8 MOTORS_ON        = 2~%uint8 MOTORS_OFF       = 3~%~%# RAPID states:~%uint8 RAPID_UNDEFINED = 1~%uint8 RAPID_STOPPED   = 2~%uint8 RAPID_RUNNING   = 3~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The EGM channel's name.~%string name~%~%# The EGM channel's active/inactive status.~%bool active~%~%# Indicator for if the EGM convergence criteria is currently fulfilled or not.~%bool egm_convergence_met~%~%# EGM client state.~%uint8 egm_client_state~%~%# Motor state.~%uint8 motor_state~%~%# RAPID execution state.~%uint8 rapid_execution_state~%~%# Utilization rate of the connected mechanical unit.~%#~%# An utilization rate > 100 mean that EGM motion references are~%# too aggressive and should be reduced.~%float64 utilization_rate~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EGMChannelState)))
  "Returns full string definition for message of type 'EGMChannelState"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of an EGM communication channel. I.e. a channel between an EGM client,~%#   in an ABB robot controller system, and an external EGM server.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# EGM client states:~%uint8 EGM_UNDEFINED = 1~%uint8 EGM_ERROR     = 2~%uint8 EGM_STOPPED   = 3~%uint8 EGM_RUNNING   = 4~%~%# Motor states:~%uint8 MOTORS_UNDEFINED = 1~%uint8 MOTORS_ON        = 2~%uint8 MOTORS_OFF       = 3~%~%# RAPID states:~%uint8 RAPID_UNDEFINED = 1~%uint8 RAPID_STOPPED   = 2~%uint8 RAPID_RUNNING   = 3~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The EGM channel's name.~%string name~%~%# The EGM channel's active/inactive status.~%bool active~%~%# Indicator for if the EGM convergence criteria is currently fulfilled or not.~%bool egm_convergence_met~%~%# EGM client state.~%uint8 egm_client_state~%~%# Motor state.~%uint8 motor_state~%~%# RAPID execution state.~%uint8 rapid_execution_state~%~%# Utilization rate of the connected mechanical unit.~%#~%# An utilization rate > 100 mean that EGM motion references are~%# too aggressive and should be reduced.~%float64 utilization_rate~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EGMChannelState>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'name))
     1
     1
     1
     1
     1
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EGMChannelState>))
  "Converts a ROS message object to a list"
  (cl:list 'EGMChannelState
    (cl:cons ':name (name msg))
    (cl:cons ':active (active msg))
    (cl:cons ':egm_convergence_met (egm_convergence_met msg))
    (cl:cons ':egm_client_state (egm_client_state msg))
    (cl:cons ':motor_state (motor_state msg))
    (cl:cons ':rapid_execution_state (rapid_execution_state msg))
    (cl:cons ':utilization_rate (utilization_rate msg))
))
