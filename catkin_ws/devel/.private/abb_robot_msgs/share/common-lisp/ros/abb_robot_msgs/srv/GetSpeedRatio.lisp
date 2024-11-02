; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude GetSpeedRatio-request.msg.html

(cl:defclass <GetSpeedRatio-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GetSpeedRatio-request (<GetSpeedRatio-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSpeedRatio-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSpeedRatio-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetSpeedRatio-request> is deprecated: use abb_robot_msgs-srv:GetSpeedRatio-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSpeedRatio-request>) ostream)
  "Serializes a message object of type '<GetSpeedRatio-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSpeedRatio-request>) istream)
  "Deserializes a message object of type '<GetSpeedRatio-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSpeedRatio-request>)))
  "Returns string type for a service object of type '<GetSpeedRatio-request>"
  "abb_robot_msgs/GetSpeedRatioRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSpeedRatio-request)))
  "Returns string type for a service object of type 'GetSpeedRatio-request"
  "abb_robot_msgs/GetSpeedRatioRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSpeedRatio-request>)))
  "Returns md5sum for a message object of type '<GetSpeedRatio-request>"
  "f0753046e29c5c4aa8b7b9ff8a761daf")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSpeedRatio-request)))
  "Returns md5sum for a message object of type 'GetSpeedRatio-request"
  "f0753046e29c5c4aa8b7b9ff8a761daf")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSpeedRatio-request>)))
  "Returns full string definition for message of type '<GetSpeedRatio-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the~%#   speed ratio for an ABB robot controller system's RAPID motions~%#   (e.g. MoveAbsJ and MoveL).~%#~%# Note:~%#   The speed ratio will be an integer in the range [0, 100] (i.e. inclusive).~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSpeedRatio-request)))
  "Returns full string definition for message of type 'GetSpeedRatio-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the~%#   speed ratio for an ABB robot controller system's RAPID motions~%#   (e.g. MoveAbsJ and MoveL).~%#~%# Note:~%#   The speed ratio will be an integer in the range [0, 100] (i.e. inclusive).~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSpeedRatio-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSpeedRatio-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSpeedRatio-request
))
;//! \htmlinclude GetSpeedRatio-response.msg.html

(cl:defclass <GetSpeedRatio-response> (roslisp-msg-protocol:ros-message)
  ((speed_ratio
    :reader speed_ratio
    :initarg :speed_ratio
    :type cl:fixnum
    :initform 0)
   (result_code
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

(cl:defclass GetSpeedRatio-response (<GetSpeedRatio-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSpeedRatio-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSpeedRatio-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetSpeedRatio-response> is deprecated: use abb_robot_msgs-srv:GetSpeedRatio-response instead.")))

(cl:ensure-generic-function 'speed_ratio-val :lambda-list '(m))
(cl:defmethod speed_ratio-val ((m <GetSpeedRatio-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:speed_ratio-val is deprecated.  Use abb_robot_msgs-srv:speed_ratio instead.")
  (speed_ratio m))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <GetSpeedRatio-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <GetSpeedRatio-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSpeedRatio-response>) ostream)
  "Serializes a message object of type '<GetSpeedRatio-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'speed_ratio)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSpeedRatio-response>) istream)
  "Deserializes a message object of type '<GetSpeedRatio-response>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'speed_ratio)) (cl:read-byte istream))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSpeedRatio-response>)))
  "Returns string type for a service object of type '<GetSpeedRatio-response>"
  "abb_robot_msgs/GetSpeedRatioResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSpeedRatio-response)))
  "Returns string type for a service object of type 'GetSpeedRatio-response"
  "abb_robot_msgs/GetSpeedRatioResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSpeedRatio-response>)))
  "Returns md5sum for a message object of type '<GetSpeedRatio-response>"
  "f0753046e29c5c4aa8b7b9ff8a761daf")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSpeedRatio-response)))
  "Returns md5sum for a message object of type 'GetSpeedRatio-response"
  "f0753046e29c5c4aa8b7b9ff8a761daf")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSpeedRatio-response>)))
  "Returns full string definition for message of type '<GetSpeedRatio-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Speed ratio.~%uint8 speed_ratio~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSpeedRatio-response)))
  "Returns full string definition for message of type 'GetSpeedRatio-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Speed ratio.~%uint8 speed_ratio~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSpeedRatio-response>))
  (cl:+ 0
     1
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSpeedRatio-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSpeedRatio-response
    (cl:cons ':speed_ratio (speed_ratio msg))
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetSpeedRatio)))
  'GetSpeedRatio-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetSpeedRatio)))
  'GetSpeedRatio-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSpeedRatio)))
  "Returns string type for a service object of type '<GetSpeedRatio>"
  "abb_robot_msgs/GetSpeedRatio")