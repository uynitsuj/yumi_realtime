; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude SetIOSignal-request.msg.html

(cl:defclass <SetIOSignal-request> (roslisp-msg-protocol:ros-message)
  ((signal
    :reader signal
    :initarg :signal
    :type cl:string
    :initform "")
   (value
    :reader value
    :initarg :value
    :type cl:string
    :initform ""))
)

(cl:defclass SetIOSignal-request (<SetIOSignal-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetIOSignal-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetIOSignal-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetIOSignal-request> is deprecated: use abb_robot_msgs-srv:SetIOSignal-request instead.")))

(cl:ensure-generic-function 'signal-val :lambda-list '(m))
(cl:defmethod signal-val ((m <SetIOSignal-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:signal-val is deprecated.  Use abb_robot_msgs-srv:signal instead.")
  (signal m))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <SetIOSignal-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:value-val is deprecated.  Use abb_robot_msgs-srv:value instead.")
  (value m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetIOSignal-request>) ostream)
  "Serializes a message object of type '<SetIOSignal-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'signal))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'signal))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'value))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetIOSignal-request>) istream)
  "Deserializes a message object of type '<SetIOSignal-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'signal) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'signal) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'value) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'value) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetIOSignal-request>)))
  "Returns string type for a service object of type '<SetIOSignal-request>"
  "abb_robot_msgs/SetIOSignalRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetIOSignal-request)))
  "Returns string type for a service object of type 'SetIOSignal-request"
  "abb_robot_msgs/SetIOSignalRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetIOSignal-request>)))
  "Returns md5sum for a message object of type '<SetIOSignal-request>"
  "0950ef5b96964231922a8ac9fd2a1318")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetIOSignal-request)))
  "Returns md5sum for a message object of type 'SetIOSignal-request"
  "0950ef5b96964231922a8ac9fd2a1318")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetIOSignal-request>)))
  "Returns full string definition for message of type '<SetIOSignal-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the~%#   value of an IO-signal defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Name of the targeted IO-signal.~%string signal~%~%# New value for the IO-signal.~%string value~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetIOSignal-request)))
  "Returns full string definition for message of type 'SetIOSignal-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to set the~%#   value of an IO-signal defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Name of the targeted IO-signal.~%string signal~%~%# New value for the IO-signal.~%string value~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetIOSignal-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'signal))
     4 (cl:length (cl:slot-value msg 'value))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetIOSignal-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetIOSignal-request
    (cl:cons ':signal (signal msg))
    (cl:cons ':value (value msg))
))
;//! \htmlinclude SetIOSignal-response.msg.html

(cl:defclass <SetIOSignal-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass SetIOSignal-response (<SetIOSignal-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetIOSignal-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetIOSignal-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<SetIOSignal-response> is deprecated: use abb_robot_msgs-srv:SetIOSignal-response instead.")))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <SetIOSignal-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetIOSignal-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetIOSignal-response>) ostream)
  "Serializes a message object of type '<SetIOSignal-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetIOSignal-response>) istream)
  "Deserializes a message object of type '<SetIOSignal-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetIOSignal-response>)))
  "Returns string type for a service object of type '<SetIOSignal-response>"
  "abb_robot_msgs/SetIOSignalResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetIOSignal-response)))
  "Returns string type for a service object of type 'SetIOSignal-response"
  "abb_robot_msgs/SetIOSignalResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetIOSignal-response>)))
  "Returns md5sum for a message object of type '<SetIOSignal-response>"
  "0950ef5b96964231922a8ac9fd2a1318")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetIOSignal-response)))
  "Returns md5sum for a message object of type 'SetIOSignal-response"
  "0950ef5b96964231922a8ac9fd2a1318")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetIOSignal-response>)))
  "Returns full string definition for message of type '<SetIOSignal-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetIOSignal-response)))
  "Returns full string definition for message of type 'SetIOSignal-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetIOSignal-response>))
  (cl:+ 0
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetIOSignal-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetIOSignal-response
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetIOSignal)))
  'SetIOSignal-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetIOSignal)))
  'SetIOSignal-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetIOSignal)))
  "Returns string type for a service object of type '<SetIOSignal>"
  "abb_robot_msgs/SetIOSignal")