; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude SetSpeedRatio-request.msg.html

(cl:defclass <SetSpeedRatio-request> (roslisp-msg-protocol:ros-message)
  ((speed_ratio
    :reader speed_ratio
    :initarg :speed_ratio
    :type cl:fixnum
    :initform 0))
)

(cl:defclass SetSpeedRatio-request (<SetSpeedRatio-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSpeedRatio-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSpeedRatio-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetSpeedRatio-request> is deprecated: use abb_robot_msgs-srv:SetSpeedRatio-request instead.")))

(cl:ensure-generic-function 'speed_ratio-val :lambda-list '(m))
(cl:defmethod speed_ratio-val ((m <SetSpeedRatio-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:speed_ratio-val is deprecated.  Use abb_robot_msgs-srv:speed_ratio instead.")
  (speed_ratio m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSpeedRatio-request>) ostream)
  "Serializes a message object of type '<SetSpeedRatio-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'speed_ratio)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSpeedRatio-request>) istream)
  "Deserializes a message object of type '<SetSpeedRatio-request>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'speed_ratio)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSpeedRatio-request>)))
  "Returns string type for a service object of type '<SetSpeedRatio-request>"
  "abb_robot_msgs/SetSpeedRatioRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSpeedRatio-request)))
  "Returns string type for a service object of type 'SetSpeedRatio-request"
  "abb_robot_msgs/SetSpeedRatioRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSpeedRatio-request>)))
  "Returns md5sum for a message object of type '<SetSpeedRatio-request>"
  "a37a98881c87fe9569621a0efb03c471")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSpeedRatio-request)))
  "Returns md5sum for a message object of type 'SetSpeedRatio-request"
  "a37a98881c87fe9569621a0efb03c471")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSpeedRatio-request>)))
  "Returns full string definition for message of type '<SetSpeedRatio-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the~%#   speed ratio for an ABB robot controller system's RAPID motions~%#   (e.g. MoveAbsJ and MoveL).~%#~%# Note:~%#   The speed ratio must be an integer in the range [0, 100] (i.e. inclusive).~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# New speed ratio.~%uint8 speed_ratio~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSpeedRatio-request)))
  "Returns full string definition for message of type 'SetSpeedRatio-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the~%#   speed ratio for an ABB robot controller system's RAPID motions~%#   (e.g. MoveAbsJ and MoveL).~%#~%# Note:~%#   The speed ratio must be an integer in the range [0, 100] (i.e. inclusive).~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# New speed ratio.~%uint8 speed_ratio~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSpeedRatio-request>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSpeedRatio-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSpeedRatio-request
    (cl:cons ':speed_ratio (speed_ratio msg))
))
;//! \htmlinclude SetSpeedRatio-response.msg.html

(cl:defclass <SetSpeedRatio-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass SetSpeedRatio-response (<SetSpeedRatio-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSpeedRatio-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSpeedRatio-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetSpeedRatio-response> is deprecated: use abb_robot_msgs-srv:SetSpeedRatio-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <SetSpeedRatio-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetSpeedRatio-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSpeedRatio-response>) ostream)
  "Serializes a message object of type '<SetSpeedRatio-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSpeedRatio-response>) istream)
  "Deserializes a message object of type '<SetSpeedRatio-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSpeedRatio-response>)))
  "Returns string type for a service object of type '<SetSpeedRatio-response>"
  "abb_robot_msgs/SetSpeedRatioResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSpeedRatio-response)))
  "Returns string type for a service object of type 'SetSpeedRatio-response"
  "abb_robot_msgs/SetSpeedRatioResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSpeedRatio-response>)))
  "Returns md5sum for a message object of type '<SetSpeedRatio-response>"
  "a37a98881c87fe9569621a0efb03c471")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSpeedRatio-response)))
  "Returns md5sum for a message object of type 'SetSpeedRatio-response"
  "a37a98881c87fe9569621a0efb03c471")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSpeedRatio-response>)))
  "Returns full string definition for message of type '<SetSpeedRatio-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSpeedRatio-response)))
  "Returns full string definition for message of type 'SetSpeedRatio-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSpeedRatio-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSpeedRatio-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSpeedRatio-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetSpeedRatio)))
  'SetSpeedRatio-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetSpeedRatio)))
  'SetSpeedRatio-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSpeedRatio)))
  "Returns string type for a service object of type '<SetSpeedRatio>"
  "abb_robot_msgs/SetSpeedRatio")