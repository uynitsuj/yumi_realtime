; Auto-generated. Do not edit!


(cl:in-package abb_egm_msgs-msg)


;//! \htmlinclude EGMState.msg.html

(cl:defclass <EGMState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (egm_channels
    :reader egm_channels
    :initarg :egm_channels
    :type (cl:vector abb_egm_msgs-msg:EGMChannelState)
   :initform (cl:make-array 0 :element-type 'abb_egm_msgs-msg:EGMChannelState :initial-element (cl:make-instance 'abb_egm_msgs-msg:EGMChannelState))))
)

(cl:defclass EGMState (<EGMState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <EGMState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'EGMState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_egm_msgs-msg:<EGMState> is deprecated: use abb_egm_msgs-msg:EGMState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <EGMState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:header-val is deprecated.  Use abb_egm_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'egm_channels-val :lambda-list '(m))
(cl:defmethod egm_channels-val ((m <EGMState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_egm_msgs-msg:egm_channels-val is deprecated.  Use abb_egm_msgs-msg:egm_channels instead.")
  (egm_channels m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <EGMState>) ostream)
  "Serializes a message object of type '<EGMState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'egm_channels))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'egm_channels))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <EGMState>) istream)
  "Deserializes a message object of type '<EGMState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'egm_channels) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'egm_channels)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'abb_egm_msgs-msg:EGMChannelState))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<EGMState>)))
  "Returns string type for a message object of type '<EGMState>"
  "abb_egm_msgs/EGMState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'EGMState)))
  "Returns string type for a message object of type 'EGMState"
  "abb_egm_msgs/EGMState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<EGMState>)))
  "Returns md5sum for a message object of type '<EGMState>"
  "a239c0f02980c16cd34f9e56fbc03ff7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'EGMState)))
  "Returns md5sum for a message object of type 'EGMState"
  "a239c0f02980c16cd34f9e56fbc03ff7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<EGMState>)))
  "Returns full string definition for message of type '<EGMState>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of all (known) EGM communication channels.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Metadata.~%std_msgs/Header header~%~%# EGM channel states.~%EGMChannelState[] egm_channels~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: abb_egm_msgs/EGMChannelState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of an EGM communication channel. I.e. a channel between an EGM client,~%#   in an ABB robot controller system, and an external EGM server.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# EGM client states:~%uint8 EGM_UNDEFINED = 1~%uint8 EGM_ERROR     = 2~%uint8 EGM_STOPPED   = 3~%uint8 EGM_RUNNING   = 4~%~%# Motor states:~%uint8 MOTORS_UNDEFINED = 1~%uint8 MOTORS_ON        = 2~%uint8 MOTORS_OFF       = 3~%~%# RAPID states:~%uint8 RAPID_UNDEFINED = 1~%uint8 RAPID_STOPPED   = 2~%uint8 RAPID_RUNNING   = 3~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The EGM channel's name.~%string name~%~%# The EGM channel's active/inactive status.~%bool active~%~%# Indicator for if the EGM convergence criteria is currently fulfilled or not.~%bool egm_convergence_met~%~%# EGM client state.~%uint8 egm_client_state~%~%# Motor state.~%uint8 motor_state~%~%# RAPID execution state.~%uint8 rapid_execution_state~%~%# Utilization rate of the connected mechanical unit.~%#~%# An utilization rate > 100 mean that EGM motion references are~%# too aggressive and should be reduced.~%float64 utilization_rate~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'EGMState)))
  "Returns full string definition for message of type 'EGMState"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of all (known) EGM communication channels.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Metadata.~%std_msgs/Header header~%~%# EGM channel states.~%EGMChannelState[] egm_channels~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: abb_egm_msgs/EGMChannelState~%#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent a subset of states~%#   of an EGM communication channel. I.e. a channel between an EGM client,~%#   in an ABB robot controller system, and an external EGM server.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# EGM client states:~%uint8 EGM_UNDEFINED = 1~%uint8 EGM_ERROR     = 2~%uint8 EGM_STOPPED   = 3~%uint8 EGM_RUNNING   = 4~%~%# Motor states:~%uint8 MOTORS_UNDEFINED = 1~%uint8 MOTORS_ON        = 2~%uint8 MOTORS_OFF       = 3~%~%# RAPID states:~%uint8 RAPID_UNDEFINED = 1~%uint8 RAPID_STOPPED   = 2~%uint8 RAPID_RUNNING   = 3~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# The EGM channel's name.~%string name~%~%# The EGM channel's active/inactive status.~%bool active~%~%# Indicator for if the EGM convergence criteria is currently fulfilled or not.~%bool egm_convergence_met~%~%# EGM client state.~%uint8 egm_client_state~%~%# Motor state.~%uint8 motor_state~%~%# RAPID execution state.~%uint8 rapid_execution_state~%~%# Utilization rate of the connected mechanical unit.~%#~%# An utilization rate > 100 mean that EGM motion references are~%# too aggressive and should be reduced.~%float64 utilization_rate~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <EGMState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'egm_channels) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <EGMState>))
  "Converts a ROS message object to a list"
  (cl:list 'EGMState
    (cl:cons ':header (header msg))
    (cl:cons ':egm_channels (egm_channels msg))
))
