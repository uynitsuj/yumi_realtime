; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude GetRobotControllerDescription-request.msg.html

(cl:defclass <GetRobotControllerDescription-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GetRobotControllerDescription-request (<GetRobotControllerDescription-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetRobotControllerDescription-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetRobotControllerDescription-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetRobotControllerDescription-request> is deprecated: use abb_robot_msgs-srv:GetRobotControllerDescription-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetRobotControllerDescription-request>) ostream)
  "Serializes a message object of type '<GetRobotControllerDescription-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetRobotControllerDescription-request>) istream)
  "Deserializes a message object of type '<GetRobotControllerDescription-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetRobotControllerDescription-request>)))
  "Returns string type for a service object of type '<GetRobotControllerDescription-request>"
  "abb_robot_msgs/GetRobotControllerDescriptionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRobotControllerDescription-request)))
  "Returns string type for a service object of type 'GetRobotControllerDescription-request"
  "abb_robot_msgs/GetRobotControllerDescriptionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetRobotControllerDescription-request>)))
  "Returns md5sum for a message object of type '<GetRobotControllerDescription-request>"
  "7c2a2ccf1dbdb360e4b26d80a3725b92")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetRobotControllerDescription-request)))
  "Returns md5sum for a message object of type 'GetRobotControllerDescription-request"
  "7c2a2ccf1dbdb360e4b26d80a3725b92")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetRobotControllerDescription-request>)))
  "Returns full string definition for message of type '<GetRobotControllerDescription-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get a~%#   (partial) description of an ABB robot controller system.~%#~%# Note:~%#   The description will be returned in a Google Protocol Buffer text format.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetRobotControllerDescription-request)))
  "Returns full string definition for message of type 'GetRobotControllerDescription-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get a~%#   (partial) description of an ABB robot controller system.~%#~%# Note:~%#   The description will be returned in a Google Protocol Buffer text format.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetRobotControllerDescription-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetRobotControllerDescription-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetRobotControllerDescription-request
))
;//! \htmlinclude GetRobotControllerDescription-response.msg.html

(cl:defclass <GetRobotControllerDescription-response> (roslisp-msg-protocol:ros-message)
  ((description
    :reader description
    :initarg :description
    :type cl:string
    :initform "")
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

(cl:defclass GetRobotControllerDescription-response (<GetRobotControllerDescription-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetRobotControllerDescription-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetRobotControllerDescription-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetRobotControllerDescription-response> is deprecated: use abb_robot_msgs-srv:GetRobotControllerDescription-response instead.")))

(cl:ensure-generic-function 'description-val :lambda-list '(m))
(cl:defmethod description-val ((m <GetRobotControllerDescription-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:description-val is deprecated.  Use abb_robot_msgs-srv:description instead.")
  (description m))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <GetRobotControllerDescription-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <GetRobotControllerDescription-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetRobotControllerDescription-response>) ostream)
  "Serializes a message object of type '<GetRobotControllerDescription-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'description))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'description))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetRobotControllerDescription-response>) istream)
  "Deserializes a message object of type '<GetRobotControllerDescription-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'description) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'description) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetRobotControllerDescription-response>)))
  "Returns string type for a service object of type '<GetRobotControllerDescription-response>"
  "abb_robot_msgs/GetRobotControllerDescriptionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRobotControllerDescription-response)))
  "Returns string type for a service object of type 'GetRobotControllerDescription-response"
  "abb_robot_msgs/GetRobotControllerDescriptionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetRobotControllerDescription-response>)))
  "Returns md5sum for a message object of type '<GetRobotControllerDescription-response>"
  "7c2a2ccf1dbdb360e4b26d80a3725b92")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetRobotControllerDescription-response)))
  "Returns md5sum for a message object of type 'GetRobotControllerDescription-response"
  "7c2a2ccf1dbdb360e4b26d80a3725b92")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetRobotControllerDescription-response>)))
  "Returns full string definition for message of type '<GetRobotControllerDescription-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Description of the robot controller system.~%string description~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetRobotControllerDescription-response)))
  "Returns full string definition for message of type 'GetRobotControllerDescription-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Description of the robot controller system.~%string description~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetRobotControllerDescription-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'description))
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetRobotControllerDescription-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetRobotControllerDescription-response
    (cl:cons ':description (description msg))
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetRobotControllerDescription)))
  'GetRobotControllerDescription-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetRobotControllerDescription)))
  'GetRobotControllerDescription-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetRobotControllerDescription)))
  "Returns string type for a service object of type '<GetRobotControllerDescription>"
  "abb_robot_msgs/GetRobotControllerDescription")