; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude TriggerWithResultCode-request.msg.html

(cl:defclass <TriggerWithResultCode-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass TriggerWithResultCode-request (<TriggerWithResultCode-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TriggerWithResultCode-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TriggerWithResultCode-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<TriggerWithResultCode-request> is deprecated: use abb_robot_msgs-srv:TriggerWithResultCode-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TriggerWithResultCode-request>) ostream)
  "Serializes a message object of type '<TriggerWithResultCode-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TriggerWithResultCode-request>) istream)
  "Deserializes a message object of type '<TriggerWithResultCode-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TriggerWithResultCode-request>)))
  "Returns string type for a service object of type '<TriggerWithResultCode-request>"
  "abb_robot_msgs/TriggerWithResultCodeRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TriggerWithResultCode-request)))
  "Returns string type for a service object of type 'TriggerWithResultCode-request"
  "abb_robot_msgs/TriggerWithResultCodeRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TriggerWithResultCode-request>)))
  "Returns md5sum for a message object of type '<TriggerWithResultCode-request>"
  "aab0cc4b756aaf08eb67ad6459b36b27")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TriggerWithResultCode-request)))
  "Returns md5sum for a message object of type 'TriggerWithResultCode-request"
  "aab0cc4b756aaf08eb67ad6459b36b27")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TriggerWithResultCode-request>)))
  "Returns full string definition for message of type '<TriggerWithResultCode-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to allow triggering a service~%#   where the response includes both a numerical result code and a human~%#   readable message.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields (none)~%#-------------------------------------------------------------------------------~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TriggerWithResultCode-request)))
  "Returns full string definition for message of type 'TriggerWithResultCode-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to allow triggering a service~%#   where the response includes both a numerical result code and a human~%#   readable message.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields (none)~%#-------------------------------------------------------------------------------~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TriggerWithResultCode-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TriggerWithResultCode-request>))
  "Converts a ROS message object to a list"
  (cl:list 'TriggerWithResultCode-request
))
;//! \htmlinclude TriggerWithResultCode-response.msg.html

(cl:defclass <TriggerWithResultCode-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass TriggerWithResultCode-response (<TriggerWithResultCode-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TriggerWithResultCode-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TriggerWithResultCode-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<TriggerWithResultCode-response> is deprecated: use abb_robot_msgs-srv:TriggerWithResultCode-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <TriggerWithResultCode-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <TriggerWithResultCode-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TriggerWithResultCode-response>) ostream)
  "Serializes a message object of type '<TriggerWithResultCode-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TriggerWithResultCode-response>) istream)
  "Deserializes a message object of type '<TriggerWithResultCode-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TriggerWithResultCode-response>)))
  "Returns string type for a service object of type '<TriggerWithResultCode-response>"
  "abb_robot_msgs/TriggerWithResultCodeResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TriggerWithResultCode-response)))
  "Returns string type for a service object of type 'TriggerWithResultCode-response"
  "abb_robot_msgs/TriggerWithResultCodeResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TriggerWithResultCode-response>)))
  "Returns md5sum for a message object of type '<TriggerWithResultCode-response>"
  "aab0cc4b756aaf08eb67ad6459b36b27")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TriggerWithResultCode-response)))
  "Returns md5sum for a message object of type 'TriggerWithResultCode-response"
  "aab0cc4b756aaf08eb67ad6459b36b27")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TriggerWithResultCode-response>)))
  "Returns full string definition for message of type '<TriggerWithResultCode-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Informational, e.g. for error messages~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TriggerWithResultCode-response)))
  "Returns full string definition for message of type 'TriggerWithResultCode-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Informational, e.g. for error messages~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TriggerWithResultCode-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TriggerWithResultCode-response>))
  "Converts a ROS message object to a list"
  (cl:list 'TriggerWithResultCode-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'TriggerWithResultCode)))
  'TriggerWithResultCode-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'TriggerWithResultCode)))
  'TriggerWithResultCode-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TriggerWithResultCode)))
  "Returns string type for a service object of type '<TriggerWithResultCode>"
  "abb_robot_msgs/TriggerWithResultCode")