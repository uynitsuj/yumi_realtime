; Auto-generated. Do not edit!


(cl:in-package abb_rapid_sm_addin_msgs-srv)


;//! \htmlinclude SetSGCommand-request.msg.html

(cl:defclass <SetSGCommand-request> (roslisp-msg-protocol:ros-message)
  ((task
    :reader task
    :initarg :task
    :type cl:string
    :initform "")
   (command
    :reader command
    :initarg :command
    :type cl:fixnum
    :initform 0)
   (target_position
    :reader target_position
    :initarg :target_position
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetSGCommand-request (<SetSGCommand-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSGCommand-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSGCommand-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-srv:<SetSGCommand-request> is deprecated: use abb_rapid_sm_addin_msgs-srv:SetSGCommand-request instead.")))

(cl:ensure-generic-function 'task-val :lambda-list '(m))
(cl:defmethod task-val ((m <SetSGCommand-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:task-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:task instead.")
  (task m))

(cl:ensure-generic-function 'command-val :lambda-list '(m))
(cl:defmethod command-val ((m <SetSGCommand-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:command-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:command instead.")
  (command m))

(cl:ensure-generic-function 'target_position-val :lambda-list '(m))
(cl:defmethod target_position-val ((m <SetSGCommand-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:target_position-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:target_position instead.")
  (target_position m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<SetSGCommand-request>)))
    "Constants for message type '<SetSGCommand-request>"
  '((:SG_COMMAND_UNKNOWN . 1)
    (:SG_COMMAND_NONE . 2)
    (:SG_COMMAND_INITIALIZE . 3)
    (:SG_COMMAND_CALIBRATE . 4)
    (:SG_COMMAND_MOVE_TO . 5)
    (:SG_COMMAND_GRIP_IN . 6)
    (:SG_COMMAND_GRIP_OUT . 7)
    (:SG_COMMAND_BLOW_ON_1 . 8)
    (:SG_COMMAND_BLOW_ON_2 . 9)
    (:SG_COMMAND_BLOW_OFF_1 . 10)
    (:SG_COMMAND_BLOW_OFF_2 . 11)
    (:SG_COMMAND_VACUUM_ON_1 . 12)
    (:SG_COMMAND_VACUUM_ON_2 . 13)
    (:SG_COMMAND_VACUUM_OFF_1 . 14)
    (:SG_COMMAND_VACUUM_OFF_2 . 15))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'SetSGCommand-request)))
    "Constants for message type 'SetSGCommand-request"
  '((:SG_COMMAND_UNKNOWN . 1)
    (:SG_COMMAND_NONE . 2)
    (:SG_COMMAND_INITIALIZE . 3)
    (:SG_COMMAND_CALIBRATE . 4)
    (:SG_COMMAND_MOVE_TO . 5)
    (:SG_COMMAND_GRIP_IN . 6)
    (:SG_COMMAND_GRIP_OUT . 7)
    (:SG_COMMAND_BLOW_ON_1 . 8)
    (:SG_COMMAND_BLOW_ON_2 . 9)
    (:SG_COMMAND_BLOW_OFF_1 . 10)
    (:SG_COMMAND_BLOW_OFF_2 . 11)
    (:SG_COMMAND_VACUUM_ON_1 . 12)
    (:SG_COMMAND_VACUUM_ON_2 . 13)
    (:SG_COMMAND_VACUUM_OFF_1 . 14)
    (:SG_COMMAND_VACUUM_OFF_2 . 15))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSGCommand-request>) ostream)
  "Serializes a message object of type '<SetSGCommand-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'task))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'task))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'command)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'target_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSGCommand-request>) istream)
  "Deserializes a message object of type '<SetSGCommand-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'task) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'task) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'command)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'target_position) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSGCommand-request>)))
  "Returns string type for a service object of type '<SetSGCommand-request>"
  "abb_rapid_sm_addin_msgs/SetSGCommandRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSGCommand-request)))
  "Returns string type for a service object of type 'SetSGCommand-request"
  "abb_rapid_sm_addin_msgs/SetSGCommandRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSGCommand-request>)))
  "Returns md5sum for a message object of type '<SetSGCommand-request>"
  "7aa352af5c8c7b889375c50673d12253")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSGCommand-request)))
  "Returns md5sum for a message object of type 'SetSGCommand-request"
  "7aa352af5c8c7b889375c50673d12253")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSGCommand-request>)))
  "Returns full string definition for message of type '<SetSGCommand-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the values~%#   of predefined RAPID variables for commanding a SmartGripper tool.~%#~%# Note:~%#   These variables are defined in each RAPID motion task, in ABB robot~%#   controller systems, that are using the RobotWare \"StateMachine Add-In\"~%#   together with a SmartGripper tool.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# Supported \"StateMachine Add-In\" SmartGripper commands:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 SG_COMMAND_UNKNOWN      = 1~%uint8 SG_COMMAND_NONE         = 2~%uint8 SG_COMMAND_INITIALIZE   = 3~%uint8 SG_COMMAND_CALIBRATE    = 4~%uint8 SG_COMMAND_MOVE_TO      = 5~%uint8 SG_COMMAND_GRIP_IN      = 6~%uint8 SG_COMMAND_GRIP_OUT     = 7~%uint8 SG_COMMAND_BLOW_ON_1    = 8~%uint8 SG_COMMAND_BLOW_ON_2    = 9~%uint8 SG_COMMAND_BLOW_OFF_1   = 10~%uint8 SG_COMMAND_BLOW_OFF_2   = 11~%uint8 SG_COMMAND_VACUUM_ON_1  = 12~%uint8 SG_COMMAND_VACUUM_ON_2  = 13~%uint8 SG_COMMAND_VACUUM_OFF_1 = 14~%uint8 SG_COMMAND_VACUUM_OFF_2 = 15~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string task~%~%# Desired SmartGripper command.~%uint8 command~%~%# Desired target position [mm] (only used for the 'SG_COMMAND_MOVE_TO' command).~%float32 target_position~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSGCommand-request)))
  "Returns full string definition for message of type 'SetSGCommand-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the values~%#   of predefined RAPID variables for commanding a SmartGripper tool.~%#~%# Note:~%#   These variables are defined in each RAPID motion task, in ABB robot~%#   controller systems, that are using the RobotWare \"StateMachine Add-In\"~%#   together with a SmartGripper tool.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Enumerations~%#-------------------------------------------------------------------------------~%# Supported \"StateMachine Add-In\" SmartGripper commands:~%#~%# Note: These enumeration values differ from the RAPID implementation~%#       values to avoid conflict with default values in ROS messages.~%uint8 SG_COMMAND_UNKNOWN      = 1~%uint8 SG_COMMAND_NONE         = 2~%uint8 SG_COMMAND_INITIALIZE   = 3~%uint8 SG_COMMAND_CALIBRATE    = 4~%uint8 SG_COMMAND_MOVE_TO      = 5~%uint8 SG_COMMAND_GRIP_IN      = 6~%uint8 SG_COMMAND_GRIP_OUT     = 7~%uint8 SG_COMMAND_BLOW_ON_1    = 8~%uint8 SG_COMMAND_BLOW_ON_2    = 9~%uint8 SG_COMMAND_BLOW_OFF_1   = 10~%uint8 SG_COMMAND_BLOW_OFF_2   = 11~%uint8 SG_COMMAND_VACUUM_ON_1  = 12~%uint8 SG_COMMAND_VACUUM_ON_2  = 13~%uint8 SG_COMMAND_VACUUM_OFF_1 = 14~%uint8 SG_COMMAND_VACUUM_OFF_2 = 15~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# The RAPID task that runs the \"StateMachine Add-In\" instance.~%string task~%~%# Desired SmartGripper command.~%uint8 command~%~%# Desired target position [mm] (only used for the 'SG_COMMAND_MOVE_TO' command).~%float32 target_position~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSGCommand-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'task))
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSGCommand-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSGCommand-request
    (cl:cons ':task (task msg))
    (cl:cons ':command (command msg))
    (cl:cons ':target_position (target_position msg))
))
;//! \htmlinclude SetSGCommand-response.msg.html

(cl:defclass <SetSGCommand-response> (roslisp-msg-protocol:ros-message)
  ((result_code
    :reader result_code
    :initarg :result_code
    :type cl:fixnum
    :initform 0)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass SetSGCommand-response (<SetSGCommand-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSGCommand-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSGCommand-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_sm_addin_msgs-srv:<SetSGCommand-response> is deprecated: use abb_rapid_sm_addin_msgs-srv:SetSGCommand-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <SetSGCommand-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:result_code-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetSGCommand-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_sm_addin_msgs-srv:message-val is deprecated.  Use abb_rapid_sm_addin_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSGCommand-response>) ostream)
  "Serializes a message object of type '<SetSGCommand-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSGCommand-response>) istream)
  "Deserializes a message object of type '<SetSGCommand-response>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) (cl:read-byte istream))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'message) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'message) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSGCommand-response>)))
  "Returns string type for a service object of type '<SetSGCommand-response>"
  "abb_rapid_sm_addin_msgs/SetSGCommandResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSGCommand-response)))
  "Returns string type for a service object of type 'SetSGCommand-response"
  "abb_rapid_sm_addin_msgs/SetSGCommandResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSGCommand-response>)))
  "Returns md5sum for a message object of type '<SetSGCommand-response>"
  "7aa352af5c8c7b889375c50673d12253")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSGCommand-response)))
  "Returns md5sum for a message object of type 'SetSGCommand-response"
  "7aa352af5c8c7b889375c50673d12253")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSGCommand-response>)))
  "Returns full string definition for message of type '<SetSGCommand-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSGCommand-response)))
  "Returns full string definition for message of type 'SetSGCommand-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSGCommand-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSGCommand-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSGCommand-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetSGCommand)))
  'SetSGCommand-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetSGCommand)))
  'SetSGCommand-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSGCommand)))
  "Returns string type for a service object of type '<SetSGCommand>"
  "abb_rapid_sm_addin_msgs/SetSGCommand")