; Auto-generated. Do not edit!


(cl:in-package abb_robot_msgs-srv)


;//! \htmlinclude GetIOSignal-request.msg.html

(cl:defclass <GetIOSignal-request> (roslisp-msg-protocol:ros-message)
  ((signal
    :reader signal
    :initarg :signal
    :type cl:string
    :initform ""))
)

(cl:defclass GetIOSignal-request (<GetIOSignal-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetIOSignal-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetIOSignal-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetIOSignal-request> is deprecated: use abb_robot_msgs-srv:GetIOSignal-request instead.")))

(cl:ensure-generic-function 'signal-val :lambda-list '(m))
(cl:defmethod signal-val ((m <GetIOSignal-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:signal-val is deprecated.  Use abb_robot_msgs-srv:signal instead.")
  (signal m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetIOSignal-request>) ostream)
  "Serializes a message object of type '<GetIOSignal-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'signal))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'signal))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetIOSignal-request>) istream)
  "Deserializes a message object of type '<GetIOSignal-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'signal) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'signal) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetIOSignal-request>)))
  "Returns string type for a service object of type '<GetIOSignal-request>"
  "abb_robot_msgs/GetIOSignalRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetIOSignal-request)))
  "Returns string type for a service object of type 'GetIOSignal-request"
  "abb_robot_msgs/GetIOSignalRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetIOSignal-request>)))
  "Returns md5sum for a message object of type '<GetIOSignal-request>"
  "edcf90cd349492ba1ca9cc7d13c4954a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetIOSignal-request)))
  "Returns md5sum for a message object of type 'GetIOSignal-request"
  "edcf90cd349492ba1ca9cc7d13c4954a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetIOSignal-request>)))
  "Returns full string definition for message of type '<GetIOSignal-request>"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the~%#   value of an IO-signal defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Name of the targeted IO-signal.~%string signal~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetIOSignal-request)))
  "Returns full string definition for message of type 'GetIOSignal-request"
  (cl:format cl:nil "#-------------------------------------------------------------------------------~%# Description:~%#   The purpose of this service definition, is to define a way to get the~%#   value of an IO-signal defined in an ABB robot controller system.~%#-------------------------------------------------------------------------------~%~%#-------------------------------------------------------------------------------~%# Service request fields~%#-------------------------------------------------------------------------------~%# Name of the targeted IO-signal.~%string signal~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetIOSignal-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'signal))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetIOSignal-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetIOSignal-request
    (cl:cons ':signal (signal msg))
))
;//! \htmlinclude GetIOSignal-response.msg.html

(cl:defclass <GetIOSignal-response> (roslisp-msg-protocol:ros-message)
  ((value
    :reader value
    :initarg :value
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

(cl:defclass GetIOSignal-response (<GetIOSignal-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetIOSignal-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetIOSignal-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name abb_robot_msgs-srv:<GetIOSignal-response> is deprecated: use abb_robot_msgs-srv:GetIOSignal-response instead.")))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <GetIOSignal-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:value-val is deprecated.  Use abb_robot_msgs-srv:value instead.")
  (value m))

(cl:ensure-generic-function 'result_code-val :lambda-list '(m))
(cl:defmethod result_code-val ((m <GetIOSignal-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:result_code-val is deprecated.  Use abb_robot_msgs-srv:result_code instead.")
  (result_code m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <GetIOSignal-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader abb_robot_msgs-srv:message-val is deprecated.  Use abb_robot_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetIOSignal-response>) ostream)
  "Serializes a message object of type '<GetIOSignal-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'value))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result_code)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'result_code)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetIOSignal-response>) istream)
  "Deserializes a message object of type '<GetIOSignal-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'value) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'value) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetIOSignal-response>)))
  "Returns string type for a service object of type '<GetIOSignal-response>"
  "abb_robot_msgs/GetIOSignalResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetIOSignal-response)))
  "Returns string type for a service object of type 'GetIOSignal-response"
  "abb_robot_msgs/GetIOSignalResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetIOSignal-response>)))
  "Returns md5sum for a message object of type '<GetIOSignal-response>"
  "edcf90cd349492ba1ca9cc7d13c4954a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetIOSignal-response)))
  "Returns md5sum for a message object of type 'GetIOSignal-response"
  "edcf90cd349492ba1ca9cc7d13c4954a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetIOSignal-response>)))
  "Returns full string definition for message of type '<GetIOSignal-response>"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Value of the IO-signal.~%string value~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetIOSignal-response)))
  "Returns full string definition for message of type 'GetIOSignal-response"
  (cl:format cl:nil "~%#-------------------------------------------------------------------------------~%# Service response fields~%#-------------------------------------------------------------------------------~%# Value of the IO-signal.~%string value~%~%# Service success/failure indicator.~%# Refer to 'abb_robot_msgs/ServiceResponses' for defined error codes.~%uint16 result_code~%~%# Status message (empty if service succeeded).~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetIOSignal-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'value))
     2
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetIOSignal-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetIOSignal-response
    (cl:cons ':value (value msg))
    (cl:cons ':result_code (result_code msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetIOSignal)))
  'GetIOSignal-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetIOSignal)))
  'GetIOSignal-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetIOSignal)))
  "Returns string type for a service object of type '<GetIOSignal>"
  "abb_robot_msgs/GetIOSignal")