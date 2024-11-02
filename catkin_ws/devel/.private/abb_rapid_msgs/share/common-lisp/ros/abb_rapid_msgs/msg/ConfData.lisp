; Auto-generated. Do not edit!


(cl:in-package abb_rapid_msgs-msg)


;//! \htmlinclude ConfData.msg.html

(cl:defclass <ConfData> (roslisp-msg-protocol:ros-message)
  ((cf1
    :reader cf1
    :initarg :cf1
    :type cl:float
    :initform 0.0)
   (cf4
    :reader cf4
    :initarg :cf4
    :type cl:float
    :initform 0.0)
   (cf6
    :reader cf6
    :initarg :cf6
    :type cl:float
    :initform 0.0)
   (cfx
    :reader cfx
    :initarg :cfx
    :type cl:float
    :initform 0.0))
)

(cl:defclass ConfData (<ConfData>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConfData>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConfData)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_rapid_msgs-msg:<ConfData> is deprecated: use abb_rapid_msgs-msg:ConfData instead.")))

(cl:ensure-generic-function 'cf1-val :lambda-list '(m))
(cl:defmethod cf1-val ((m <ConfData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:cf1-val is deprecated.  Use abb_rapid_msgs-msg:cf1 instead.")
  (cf1 m))

(cl:ensure-generic-function 'cf4-val :lambda-list '(m))
(cl:defmethod cf4-val ((m <ConfData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:cf4-val is deprecated.  Use abb_rapid_msgs-msg:cf4 instead.")
  (cf4 m))

(cl:ensure-generic-function 'cf6-val :lambda-list '(m))
(cl:defmethod cf6-val ((m <ConfData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:cf6-val is deprecated.  Use abb_rapid_msgs-msg:cf6 instead.")
  (cf6 m))

(cl:ensure-generic-function 'cfx-val :lambda-list '(m))
(cl:defmethod cfx-val ((m <ConfData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_rapid_msgs-msg:cfx-val is deprecated.  Use abb_rapid_msgs-msg:cfx instead.")
  (cfx m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConfData>) ostream)
  "Serializes a message object of type '<ConfData>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'cf1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'cf4))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'cf6))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'cfx))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConfData>) istream)
  "Deserializes a message object of type '<ConfData>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'cf1) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'cf4) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'cf6) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'cfx) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConfData>)))
  "Returns string type for a message object of type '<ConfData>"
  "abb_rapid_msgs/ConfData")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConfData)))
  "Returns string type for a message object of type 'ConfData"
  "abb_rapid_msgs/ConfData")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConfData>)))
  "Returns md5sum for a message object of type '<ConfData>"
  "21f1129f94cd63dd0cdf33be2bddc75e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConfData)))
  "Returns md5sum for a message object of type 'ConfData"
  "21f1129f94cd63dd0cdf33be2bddc75e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConfData>)))
  "Returns full string definition for message of type '<ConfData>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"confdata\" (robot configuration data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Rotating axis:~%# - The current quadrant of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%float32 cf1~%float32 cf4~%float32 cf6~%~%# Rotating axis:~%# - For serial link robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For SCARA robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 1.~%# - For 7-axis robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For paint robots, the current quadrant of axis 5,~%#   expressed as a positive or negative integer.~%# - For other robots, using the current quadrant of axis 2,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 2,~%#   expressed as a positive or negative integer.~%float32 cfx~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConfData)))
  "Returns full string definition for message of type 'ConfData"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this message definition, is to represent the~%#   RAPID data type \"confdata\" (robot configuration data).~%#~%# Note:~%#   Please see the \"Technical reference manual - RAPID Instructions,~%#   Functions and Data types\" for more details.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Message fields~%#-------------------------------------------------------------------------------~%# Rotating axis:~%# - The current quadrant of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 1, 4 and 6,~%#   expressed as a positive or negative integer.~%float32 cf1~%float32 cf4~%float32 cf6~%~%# Rotating axis:~%# - For serial link robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For SCARA robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 1.~%# - For 7-axis robots, the current robot configuration,~%#   expressed as an integer in the range from 0 to 7.~%# - For paint robots, the current quadrant of axis 5,~%#   expressed as a positive or negative integer.~%# - For other robots, using the current quadrant of axis 2,~%#   expressed as a positive or negative integer.~%# Linear axis:~%# - The current meter interval of axis 2,~%#   expressed as a positive or negative integer.~%float32 cfx~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConfData>))
  (cl:+ 0
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConfData>))
  "Converts a ROS message object to a list"
  (cl:list 'ConfData
    (cl:cons ':cf1 (cf1 msg))
    (cl:cons ':cf4 (cf4 msg))
    (cl:cons ':cf6 (cf6 msg))
    (cl:cons ':cfx (cfx msg))
))
